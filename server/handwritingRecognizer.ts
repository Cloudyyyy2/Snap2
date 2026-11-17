import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { pipeline, env, RawImage } from '@huggingface/transformers';
import { PSM } from 'tesseract.js';
import {
  cropDarkBordersAndMargins,
  removeRuledLines,
  filterTextComponents,
  computeRecommendedDpiScale,
} from '../src/lib/ocr/handwriting/cleanup';
import {
  toGrayscale,
  autoInvertIfDarkMode,
  flattenBackground,
  applyMedianFilter3x3,
  applyGaussianBlur,
  applyBilateralSmoothing,
  estimateNoiseLevel,
  enhanceInkContrast,
  binarizeSauvola,
  binarizeOtsu,
  applyMorphologicalOp,
} from '../src/lib/ocr/preprocess';
import {
  correctDocumentPerspective,
  deskewGrayscale,
} from '../src/lib/ocr/geometry';
import { segmentLayoutRegions } from '../src/lib/ocr/handwriting/layout';
import {
  segmentLines,
  splitWideLineBox,
  cropLine,
  CroppedLine,
  LineBox,
} from '../src/lib/ocr/handwriting/lines';
import { detectHandwriting, HandwritingDetectionResult } from '../src/lib/ocr/handwriting/detect';
import {
  postprocessHandwrittenLines,
  detectAndCleanHallucination,
} from '../src/lib/ocr/postprocess';
import { getCompactLexicon } from '../src/lib/ocr/compactLexicon';
import { correctTextWithLexicon } from '../src/lib/ocr/wordDataset';
import { twoPassContourVoting } from '../src/lib/ocr/preprocess';
import { isPrintedModelAvailable, recognizePrintedBuffer } from './printedRecognizer';

export type OcrProfile = 'auto' | 'printed' | 'handwriting';
export type DetectedStyle = 'printed' | 'handwriting' | 'mixed';
export type RecognizerEngine = 'trocr-host' | 'tesseract-host' | 'hybrid-host' | 'vision-fallback';

export interface LocalOcrLineResult {
  text: string;
  confidence: number;
  bbox: { x0: number; y0: number; x1: number; y1: number };
}

export interface LocalOcrResponse {
  text: string;
  lines: LocalOcrLineResult[];
  profile: OcrProfile;
  detectedStyle: DetectedStyle;
  detection: HandwritingDetectionResult;
  recognizer: RecognizerEngine;
  qualityScore: number;
  skewAngle?: number;
  darkInverted?: boolean;
  perspectiveWarped?: boolean;
  bordersCropped?: boolean;
  dpiScale?: number;
  appliedSteps: string[];
  regionsCount: number;
  processedPreview?: string;
  droppedHallucinations?: number;
  exportDir?: string;
  accuracyScore?: number;
  verifiedWords?: number;
  totalWords?: number;
}

const BASE_MODEL_DIR_NAME = 'trocr-base-handwritten';
const SMALL_MODEL_DIR_NAME = 'trocr-small-handwritten';
const MODELS_ROOT = path.resolve(process.cwd(), 'public/models');

const REQUIRED_MODEL_FILES = [
  'config.json',
  'preprocessor_config.json',
  'tokenizer.json',
  'onnx/encoder_model_quantized.onnx',
  'onnx/decoder_model_merged_quantized.onnx',
];

let recognizerPromise: Promise<any> | null = null;
let loadedModelName: string | null = null;

function hasAllRequiredFiles(dir: string): boolean {
  return REQUIRED_MODEL_FILES.every((rel) => {
    const full = path.join(dir, rel);
    return fs.existsSync(full) && fs.statSync(full).size > 0;
  });
}

export function getActiveHandwritingModelName(
  customModelDir?: string
): 'trocr-base-handwritten' | 'trocr-small-handwritten' | 'custom' | 'missing' {
  if (customModelDir) {
    return hasAllRequiredFiles(customModelDir) ? 'custom' : 'missing';
  }
  if (hasAllRequiredFiles(path.join(MODELS_ROOT, BASE_MODEL_DIR_NAME))) {
    return BASE_MODEL_DIR_NAME;
  }
  if (hasAllRequiredFiles(path.join(MODELS_ROOT, SMALL_MODEL_DIR_NAME))) {
    return SMALL_MODEL_DIR_NAME;
  }
  return 'missing';
}

export function getModelDir(): string {
  const active = getActiveHandwritingModelName();
  return path.join(
    MODELS_ROOT,
    active === 'trocr-small-handwritten' ? SMALL_MODEL_DIR_NAME : BASE_MODEL_DIR_NAME
  );
}

export function isHandwritingModelAvailable(customModelDir?: string): boolean {
  return getActiveHandwritingModelName(customModelDir) !== 'missing';
}

export async function getHandwritingRecognizer() {
  const activeName = getActiveHandwritingModelName();
  if (activeName === 'missing' || activeName === 'custom') {
    throw new Error(
      `Local handwriting model missing in ${MODELS_ROOT}. Run "npm run models:handwriting" first.`
    );
  }

  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.localModelPath = MODELS_ROOT;

  if (!recognizerPromise || loadedModelName !== activeName) {
    loadedModelName = activeName;
    recognizerPromise = pipeline('image-to-text', activeName, {
      dtype: 'q8',
    }).catch((err) => {
      recognizerPromise = null;
      loadedModelName = null;
      throw err;
    });
  }

  return recognizerPromise;
}

function grayscaleToRgbRawImage(gray: Uint8ClampedArray, width: number, height: number): RawImage {
  const rgb = new Uint8ClampedArray(width * height * 3);
  for (let i = 0; i < gray.length; i++) {
    const v = gray[i];
    const j = i * 3;
    rgb[j] = v;
    rgb[j + 1] = v;
    rgb[j + 2] = v;
  }
  return new RawImage(rgb, width, height, 3);
}

function scoreRecognizedLine(text: string): number {
  const { text: clean, isHallucination } = detectAndCleanHallucination(text);
  if (isHallucination || !clean) return 0.15;

  const words = clean
    .toLowerCase()
    .split(/\s+/)
    .map((w) => w.replace(/[^a-z0-9]/g, ''))
    .filter(Boolean);

  if (words.length === 0) return 0.2;

  const lexicon = getCompactLexicon();
  let known = 0;
  for (const w of words) {
    if (lexicon.has(w) || /^\d+(?:am|pm)?$/.test(w)) {
      known++;
    }
  }
  const lexRatio = known / words.length;
  return Number(Math.min(0.92, 0.52 + lexRatio * 0.4).toFixed(2));
}

function scoreTextPlausibility(text: string, qualityScore: number): number {
  const trimmed = text.trim();
  if (!trimmed) return 0;

  const rawTokens = trimmed.split(/\s+/).filter(Boolean);
  if (rawTokens.length === 0) return 0;

  const lexicon = getCompactLexicon();
  let lexiconOrNumericHits = 0;
  let alphaNumericTokens = 0;
  let weirdSymbolTokens = 0;

  for (const rawTok of rawTokens) {
    const cleaned = rawTok.toLowerCase().replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, '');
    if (!cleaned) {
      weirdSymbolTokens++;
      continue;
    }
    alphaNumericTokens++;
    if (
      lexicon.has(cleaned) ||
      /^\d+(?:[.,:/-]\d+)*(?:am|pm|k|m)?$/.test(cleaned) ||
      (/^[a-z]{3,}$/.test(cleaned) && /[aeiouy]/.test(cleaned))
    ) {
      lexiconOrNumericHits++;
    }
  }

  if (alphaNumericTokens === 0) return 0.05;

  const lexRatio = lexiconOrNumericHits / rawTokens.length;
  const cleanRatio = Math.max(0, 1 - weirdSymbolTokens / rawTokens.length);
  const clampedQuality = Math.max(0, Math.min(1, qualityScore));

  return Number((clampedQuality * 0.5 + lexRatio * 0.35 + cleanRatio * 0.15).toFixed(3));
}

/**
 * Runs TrOCR-base-handwritten sequentially line-by-line (chunking ultra-wide lines
 * cleanly so 384x384 input resizing never squashes words beyond recognition).
 */
export async function recognizeLines(
  lineCrops: CroppedLine[]
): Promise<Array<{ text: string; confidence: number }>> {
  const rec = await getHandwritingRecognizer();
  const results: Array<{ text: string; confidence: number }> = [];

  for (const crop of lineCrops) {
    const rawImg = grayscaleToRgbRawImage(crop.data, crop.width, crop.height);
    const out = await rec(rawImg, {
      max_new_tokens: 48,
      no_repeat_ngram_size: 3,
      repetition_penalty: 1.2,
    });
    const first = Array.isArray(out) ? out[0] : out;
    const text = typeof first?.generated_text === 'string' ? first.generated_text.trim() : '';

    const confidence =
      typeof first?.score === 'number' ? first.score : scoreRecognizedLine(text);

    results.push({ text, confidence });
  }

  return results;
}

async function saveGrayscalePng(
  filePath: string,
  gray: Uint8ClampedArray,
  width: number,
  height: number
): Promise<void> {
  await sharp(Buffer.from(gray), {
    raw: { width, height, channels: 1 },
  })
    .png()
    .toFile(filePath);
}

async function encodePreviewDataUrl(
  gray: Uint8ClampedArray,
  width: number,
  height: number
): Promise<string> {
  const buf = await sharp(Buffer.from(gray), {
    raw: { width, height, channels: 1 },
  })
    .resize({ width: 960, withoutEnlargement: true })
    .png({ compressionLevel: 8 })
    .toBuffer();
  return `data:image/png;base64,${buf.toString('base64')}`;
}

export function decodeBase64Image(imageInput: string): Buffer {
  const matches = imageInput.match(/^data:(.+);base64,(.+)$/);
  const base64 = matches ? matches[2] : imageInput;
  return Buffer.from(base64, 'base64');
}

function cropRegionWithPadding(
  gray: Uint8ClampedArray,
  w: number,
  h: number,
  box: { x0: number; y0: number; x1: number; y1: number },
  pad = 16
): { data: Uint8ClampedArray; width: number; height: number } {
  const rw = Math.max(1, box.x1 - box.x0 + 1 + pad * 2);
  const rh = Math.max(1, box.y1 - box.y0 + 1 + pad * 2);
  const out = new Uint8ClampedArray(rw * rh);
  out.fill(255);

  for (let y = box.y0; y <= box.y1; y++) {
    if (y < 0 || y >= h) continue;
    const dstY = y - box.y0 + pad;
    const srcOff = y * w;
    const dstOff = dstY * rw;
    for (let x = box.x0; x <= box.x1; x++) {
      if (x < 0 || x >= w) continue;
      out[dstOff + (x - box.x0 + pad)] = gray[srcOff + x];
    }
  }

  return { data: out, width: rw, height: rh };
}

function maskBinaryOutsideBox(
  binary: Uint8ClampedArray,
  w: number,
  h: number,
  box: { x0: number; y0: number; x1: number; y1: number }
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h).fill(255);
  const y0 = Math.max(0, box.y0);
  const y1 = Math.min(h - 1, box.y1);
  const x0 = Math.max(0, box.x0);
  const x1 = Math.min(w - 1, box.x1);
  for (let y = y0; y <= y1; y++) {
    const rowOff = y * w;
    for (let x = x0; x <= x1; x++) {
      out[rowOff + x] = binary[rowOff + x];
    }
  }
  return out;
}

function verticalOverlapRatio(
  a: { y0: number; y1: number },
  b: { y0: number; y1: number }
): number {
  const inter = Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0));
  if (inter <= 0) return 0;
  const minH = Math.max(1, Math.min(a.y1 - a.y0, b.y1 - b.y0));
  return inter / minH;
}

/**
 * Full 4-Stage Offline Image Processing + Dual Recognition Pipeline:
 * 1. Luminance & Contrast Normalization (grayscale collapse, polarity inversion, Retinex flattening, Otsu/Sauvola)
 * 2. Geometric & Spatial Correction (perspective quad dewarping, deskewing, 300+ DPI scaling)
 * 3. Noise Removal & Edge Preservation (3x3 median, bilateral smoothing, border cropping, morphological ops)
 * 4. Layout & Region Segmentation (XY-cut text regions, ruled-line removal, line/word segmentation)
 * -> Dual Offline Recognizers (Tesseract LSTM for printed, TrOCR for handwriting, hybrid for mixed pages)
 */
export async function processImageWithLocalOcr(params: {
  imageBuffer: Buffer;
  ocrProfile?: OcrProfile;
  exportStages?: boolean;
  exportRootDir?: string;
}): Promise<LocalOcrResponse> {
  const ocrProfile: OcrProfile = params.ocrProfile || 'auto';
  const exportStages = params.exportStages !== false;
  const exportRoot = params.exportRootDir || path.resolve(process.cwd(), 'ocr-exports');
  const appliedSteps: string[] = ['grayscale-luminance'];

  // Decode & EXIF-orient upright RGBA, then convert via BT.601 luminance + white-paper alpha compositing
  const { data: rawRgbaBuffer, info } = await sharp(params.imageBuffer)
    .rotate()
    .resize({ width: 1800, withoutEnlargement: true })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  let w = info.width;
  let h = info.height;
  const rawGray = toGrayscale(rawRgbaBuffer, w, h, 4);

  // Stage 1: Luminance & Polarity Normalization
  const { data: polarNormalized, inverted: darkInverted } = autoInvertIfDarkMode(rawGray, w, h);
  if (darkInverted) appliedSteps.push('color-inversion');

  // Quick local contrast edge-energy estimator to verify perspective warp / border crop never discards the note's text
  const measureStrokeEnergy = (buf: Uint8ClampedArray, bw: number, bh: number): number => {
    let edges = 0;
    const step = Math.max(1, Math.floor(Math.min(bw, bh) / 140));
    for (let y = 1; y < bh - 1; y += step) {
      const off = y * bw;
      for (let x = 1; x < bw - 1; x += step) {
        const dx = Math.abs(buf[off + x + 1] - buf[off + x - 1]);
        const dy = Math.abs(buf[off + bw + x] - buf[off - bw + x]);
        if (dx > 28 || dy > 28) edges++;
      }
    }
    return edges;
  };

  // Stage 2a & 3c: Perspective Quad Dewarping & Dark Border Cropping BEFORE illumination flattening
  // so dark desk surfaces retain contrast against the paper sheet rather than normalizing to 255
  let workGray = polarNormalized;
  let perspectiveWarped = false;
  let bordersCropped = false;
  const preGeomEnergy = measureStrokeEnergy(workGray, w, h);

  const persp = correctDocumentPerspective(workGray, w, h);
  if (persp.warped) {
    const warpedEnergy = measureStrokeEnergy(persp.data, persp.width, persp.height);
    // Keep perspective warp only if it preserves stroke/text detail (never crop away text into blank margin)
    if (preGeomEnergy < 40 || warpedEnergy >= preGeomEnergy * 0.25) {
      workGray = persp.data;
      w = persp.width;
      h = persp.height;
      perspectiveWarped = true;
      appliedSteps.push('perspective-dewarp');
    }
  }

  if (!perspectiveWarped) {
    const preCropEnergy = measureStrokeEnergy(workGray, w, h);
    const croppedDoc = cropDarkBordersAndMargins(workGray, w, h);
    if (croppedDoc.cropped) {
      const croppedEnergy = measureStrokeEnergy(
        croppedDoc.data,
        croppedDoc.width,
        croppedDoc.height
      );
      if (preCropEnergy < 40 || croppedEnergy >= preCropEnergy * 0.35) {
        workGray = croppedDoc.data;
        w = croppedDoc.width;
        h = croppedDoc.height;
        bordersCropped = true;
        appliedSteps.push('border-crop');
      }
    }
  }

  // Stage 1b: Retinex Illumination Flattening across the isolated paper sheet
  workGray = flattenBackground(workGray, w, h);
  appliedSteps.push('illumination-flatten');

  // Stage 3a & 3b: Salt-and-Pepper Median Filtering + Adaptive Gaussian / Bilateral Smoothing
  const measuredNoise = estimateNoiseLevel(workGray, w, h);
  workGray = applyMedianFilter3x3(workGray, w, h);
  appliedSteps.push('median-3x3');

  if (measuredNoise >= 4.0) {
    workGray = applyGaussianBlur(workGray, w, h, 0.65, 1);
    appliedSteps.push('gaussian-denoise');
  }

  if (w >= 260 && h >= 180) {
    workGray = applyBilateralSmoothing(workGray, w, h, 1.8, 22);
    appliedSteps.push('bilateral-smooth');
  }

  const getSauvolaParams = (cw: number, ch: number) => ({
    radius: Math.max(14, Math.min(36, Math.round(Math.min(cw, ch) / 28))),
    k: 0.16,
  });

  // Step 1: Initial fast binarization to capture slant angle from ruling lines and text baselines
  const spInit = getSauvolaParams(w, h);
  const rawBinaryForDeskew = binarizeSauvola(workGray, w, h, spInit.radius, spInit.k);

  // Step 2: Deskew BEFORE ruled-line removal so tilted ruling lines become horizontal (up to ±28°)
  const deskew = deskewGrayscale(workGray, rawBinaryForDeskew, w, h, {
    minCorrectionDeg: 0.4,
    maxAngleDeg: 28,
  });
  const skewAngle = deskew.angleDeg;
  if (deskew.corrected) {
    workGray = deskew.data;
    appliedSteps.push(`deskew(${skewAngle}deg)`);
  }

  // Step 3: Ruled-line removal on the leveled horizontal paper sheet
  let noRules = removeRuledLines(workGray, w, h);
  appliedSteps.push('ruled-lines-removed');

  // Step 4: Re-binarize on the leveled, rule-free paper
  let sp = getSauvolaParams(w, h);
  let binary = binarizeSauvola(noRules, w, h, sp.radius, sp.k);
  let cleanBinary = w >= 280 && h >= 200 ? filterTextComponents(binary, w, h) : binary;

  const workGrayW = w;
  const workGrayH = h;

  // Stage 2c: Smart DPI Resolution Scaling for small UI fonts / receipts (~300+ DPI target)
  const dpiScale = computeRecommendedDpiScale(cleanBinary, w, h);
  if (dpiScale >= 1.5 && Math.round(w * dpiScale) <= 2400 && Math.round(h * dpiScale) <= 2400) {
    const targetW = Math.round(w * dpiScale);
    const targetH = Math.round(h * dpiScale);
    const upscaled = await sharp(Buffer.from(noRules), {
      raw: { width: w, height: h, channels: 1 },
    })
      .resize(targetW, targetH, { kernel: sharp.kernel.lanczos3 })
      .grayscale()
      .raw()
      .toBuffer({ resolveWithObject: true });

    noRules = new Uint8ClampedArray(upscaled.data.subarray(0, upscaled.info.width * upscaled.info.height));
    w = upscaled.info.width;
    h = upscaled.info.height;
    sp = getSauvolaParams(w, h);
    binary = binarizeSauvola(noRules, w, h, sp.radius, sp.k);
    cleanBinary = w >= 280 && h >= 200 ? filterTextComponents(binary, w, h) : binary;
    appliedSteps.push(`dpi-upscale(${dpiScale}x)`);
  }

  // Stage 3b: Two-Pass Multi-Scale Contour Consensus Voting
  const contourVote = twoPassContourVoting(noRules, w, h);
  let blendedBinary = contourVote.binary;
  let regionBinary =
    w >= 280 && h >= 200 ? filterTextComponents(blendedBinary, w, h) : blendedBinary;

  // Stage 3c: Text Bounding Box Auto-Shrink (trims dead empty margins, e.g. 75% empty ruled paper below note)
  let textMinX = w;
  let textMaxX = 0;
  let textMinY = h;
  let textMaxY = 0;
  let validTextPixels = 0;

  for (let y = 0; y < h; y++) {
    const rowOff = y * w;
    for (let x = 0; x < w; x++) {
      if (regionBinary[rowOff + x] === 0) {
        validTextPixels++;
        if (x < textMinX) textMinX = x;
        if (x > textMaxX) textMaxX = x;
        if (y < textMinY) textMinY = y;
        if (y > textMaxY) textMaxY = y;
      }
    }
  }

  // If text cluster is localized and surrounded by large empty margin (> 25% empty height or width):
  if (validTextPixels >= 25 && textMaxX > textMinX + 15 && textMaxY > textMinY + 10) {
    const textSpanY = textMaxY - textMinY;
    const textSpanX = textMaxX - textMinX;

    if (textSpanY < h * 0.75 || textSpanX < w * 0.82) {
      const padX = Math.max(32, Math.round(textSpanX * 0.14));
      const padY = Math.max(26, Math.round(textSpanY * 0.16));

      const cropX0 = Math.max(0, textMinX - padX);
      const cropY0 = Math.max(0, textMinY - padY);
      const cropX1 = Math.min(w - 1, textMaxX + padX);
      const cropY1 = Math.min(h - 1, textMaxY + padY);

      const trimmedW = cropX1 - cropX0 + 1;
      const trimmedH = cropY1 - cropY0 + 1;

      if (trimmedW >= 80 && trimmedH >= 40 && (trimmedW < w * 0.95 || trimmedH < h * 0.95)) {
        const cropBuf = (src: Uint8ClampedArray): Uint8ClampedArray => {
          const out = new Uint8ClampedArray(trimmedW * trimmedH).fill(255);
          for (let cy = cropY0; cy <= cropY1; cy++) {
            const srcOff = cy * w;
            const dstOff = (cy - cropY0) * trimmedW;
            for (let cx = cropX0; cx <= cropX1; cx++) {
              out[dstOff + (cx - cropX0)] = src[srcOff + cx];
            }
          }
          return out;
        };

        workGray = cropBuf(workGray);
        noRules = cropBuf(noRules);
        blendedBinary = cropBuf(blendedBinary);
        regionBinary = cropBuf(regionBinary);
        cleanBinary = cropBuf(cleanBinary);
        w = trimmedW;
        h = trimmedH;
        appliedSteps.push(`auto-bbox-shrink(${w}x${h})`);
      }
    }
  }

  // Morphological closing bridges broken faint pencil/pen strokes for line segmentation
  const closedForSeg = applyMorphologicalOp(regionBinary, w, h, 'close', 1);
  appliedSteps.push('two-pass-contour-voting', 'morph-close');

  const contrastGray = enhanceInkContrast(noRules, w, h);
  appliedSteps.push('ink-contrast-stretch');

  // Stage 4: Layout Region Segmentation & Style Detection
  // Note: detectHandwriting uses un-closed cleanBinary so morph-close never bridges adjacent printed glyphs into fake cursive!
  const regions = segmentLayoutRegions(regionBinary, w, h);
  appliedSteps.push(`layout-regions(${regions.length})`);

  const detection = detectHandwriting(cleanBinary, w, h);
  const initialStyle: 'printed' | 'handwriting' = detection.isHandwriting
    ? 'handwriting'
    : 'printed';

  const handwritingModelReady = isHandwritingModelAvailable();
  const printedModelReady = isPrintedModelAvailable();
  const processedPreview = await encodePreviewDataUrl(contrastGray, w, h);

  let runExportDir: string | undefined;
  if (exportStages) {
    runExportDir = path.join(exportRoot, `run-${Date.now()}`);
    fs.mkdirSync(runExportDir, { recursive: true });
    await saveGrayscalePng(
      path.join(runExportDir, '03b_background_flattened.png'),
      workGray,
      workGrayW,
      workGrayH
    );
    await saveGrayscalePng(path.join(runExportDir, '03c_ruled_lines_removed.png'), noRules, w, h);
    await saveGrayscalePng(path.join(runExportDir, '04_contrast_enhanced.png'), contrastGray, w, h);
    await saveGrayscalePng(path.join(runExportDir, '05_adaptive_binary.png'), regionBinary, w, h);
  }

  // Blank / covered lens / low-ink edge guard (prevents TrOCR hallucinations on empty frames)
  let totalInkPixels = 0;
  for (let i = 0; i < regionBinary.length; i++) {
    if (regionBinary[i] === 0) totalInkPixels++;
  }
  if (totalInkPixels < 30 || regions.length === 0) {
    appliedSteps.push('no-ink-detected');
    if (runExportDir) {
      fs.writeFileSync(
        path.join(runExportDir, 'extracted_text.txt'),
        `Profile: ${ocrProfile}\nPipeline steps: ${appliedSteps.join(' -> ')}\nNo readable ink detected.`,
        'utf8'
      );
    }
    return {
      text: '',
      lines: [],
      profile: ocrProfile,
      detectedStyle: initialStyle,
      detection,
      recognizer: printedModelReady
        ? 'tesseract-host'
        : handwritingModelReady
          ? 'trocr-host'
          : 'vision-fallback',
      qualityScore: 0,
      skewAngle,
      darkInverted,
      perspectiveWarped,
      bordersCropped,
      dpiScale,
      appliedSteps,
      regionsCount: regions.length,
      processedPreview,
      exportDir: runExportDir,
    };
  }

  // Fallback immediately if neither local model exists
  if (!handwritingModelReady && !printedModelReady) {
    return {
      text: '',
      lines: [],
      profile: ocrProfile,
      detectedStyle: initialStyle,
      detection,
      recognizer: 'vision-fallback',
      qualityScore: 0,
      skewAngle,
      darkInverted,
      perspectiveWarped,
      bordersCropped,
      dpiScale,
      appliedSteps,
      regionsCount: regions.length,
      processedPreview,
      exportDir: runExportDir,
    };
  }

  const runOfflinePrintedOnBoxes = async (
    targetRegions: Array<{ x0: number; y0: number; x1: number; y1: number }>
  ): Promise<{
    text: string;
    lines: LocalOcrLineResult[];
    qualityScore: number;
  }> => {
    const pad = 18;
    const allLines: LocalOcrLineResult[] = [];
    const regionTexts: string[] = [];

    for (const reg of targetRegions) {
      const crop = cropRegionWithPadding(contrastGray, w, h, reg, pad);
      const res = await recognizePrintedBuffer({
        grayOrBinary: crop.data,
        width: crop.width,
        height: crop.height,
        psm: targetRegions.length > 1 || regions.length > 1 ? PSM.SINGLE_BLOCK : PSM.AUTO,
      });
      if (res.text.trim()) {
        regionTexts.push(res.text.trim());
        for (const l of res.lines) {
          allLines.push({
            text: l.text,
            confidence: l.confidence,
            bbox: {
              x0: Math.max(0, reg.x0 + l.bbox.x0 - pad),
              y0: Math.max(0, reg.y0 + l.bbox.y0 - pad),
              x1: Math.min(w - 1, reg.x0 + l.bbox.x1 - pad),
              y1: Math.min(h - 1, reg.y0 + l.bbox.y1 - pad),
            },
          });
        }
      }
    }

    const joined = regionTexts.join('\n').trim();
    const avgConf =
      allLines.length > 0
        ? allLines.reduce((acc, l) => acc + l.confidence, 0) / allLines.length
        : 0;
    return {
      text: joined,
      lines: allLines,
      qualityScore: Number(avgConf.toFixed(2)),
    };
  };

  const runOfflineHandwritingOnLines = async (
    targetLineBoxes: LineBox[],
    exportLineImages = false
  ): Promise<{
    text: string;
    lines: LocalOcrLineResult[];
    qualityScore: number;
    droppedHallucinations: number;
    rawCount: number;
  }> => {
    if (!handwritingModelReady || targetLineBoxes.length === 0) {
      return { text: '', lines: [], qualityScore: 0, droppedHallucinations: 0, rawCount: 0 };
    }

    const chunkPlans = targetLineBoxes.map((box) => ({
      lineBox: box,
      chunks: splitWideLineBox(closedForSeg, w, h, box, 5.0),
    }));

    const flatChunks: CroppedLine[] = [];
    for (const plan of chunkPlans) {
      for (const chunkBox of plan.chunks) {
        flatChunks.push(cropLine(contrastGray, w, h, chunkBox, 16));
      }
    }

    if (exportLineImages && runExportDir) {
      for (let i = 0; i < flatChunks.length; i++) {
        const idx = String(i + 1).padStart(2, '0');
        await saveGrayscalePng(
          path.join(runExportDir, `06_line_${idx}.png`),
          flatChunks[i].data,
          flatChunks[i].width,
          flatChunks[i].height
        );
      }
    }

    const recognizedChunks = await recognizeLines(flatChunks);
    let cursor = 0;
    const rawLines = chunkPlans.map((plan) => {
      const count = plan.chunks.length;
      const slice = recognizedChunks.slice(cursor, cursor + count);
      cursor += count;
      const joinedText = slice
        .map((s) => s.text)
        .filter(Boolean)
        .join(' ')
        .trim();
      const avgConf =
        slice.length > 0
          ? slice.reduce((acc, s) => acc + s.confidence, 0) / slice.length
          : 0;
      return {
        text: joinedText,
        confidence: Number(avgConf.toFixed(2)),
        bbox: plan.lineBox,
      };
    });

    const cleanedLines = postprocessHandwrittenLines(rawLines);
    const droppedHallucinations = Math.max(0, rawLines.length - cleanedLines.length);

    const finalLines: LocalOcrLineResult[] = cleanedLines.map((l) => ({
      text: l.text,
      confidence: l.confidence,
      bbox: l.bbox || { x0: 0, y0: 0, x1: w, y1: h },
    }));

    const fullText = finalLines.map((l) => l.text).join('\n');
    const meanLineConf =
      finalLines.length > 0
        ? finalLines.reduce((acc, l) => acc + l.confidence, 0) / finalLines.length
        : 0;
    const survivalRatio = rawLines.length > 0 ? finalLines.length / rawLines.length : 0;
    const qualityScore = Number(
      (finalLines.length > 0 ? meanLineConf * (0.5 + 0.5 * survivalRatio) : 0).toFixed(2)
    );

    return {
      text: fullText,
      lines: finalLines,
      qualityScore,
      droppedHallucinations,
      rawCount: targetLineBoxes.length,
    };
  };

  const getOrderedLineBoxesForRegions = (
    targetRegions: Array<{ x0: number; y0: number; x1: number; y1: number }>
  ): LineBox[] => {
    const hasSideBySideColumns =
      targetRegions.length > 1 &&
      targetRegions.some((a, idx) =>
        targetRegions.some(
          (b, jdx) =>
            idx !== jdx &&
            (a.x1 < b.x0 - 20 || b.x1 < a.x0 - 20) &&
            Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0) > 20
        )
      );

    if (hasSideBySideColumns || targetRegions.length === 1) {
      const boxes: LineBox[] = [];
      for (const reg of targetRegions) {
        const regBin = maskBinaryOutsideBox(closedForSeg, w, h, reg);
        boxes.push(...segmentLines(regBin, w, h));
      }
      if (boxes.length > 0) return boxes;
    }
    return segmentLines(closedForSeg, w, h);
  };

  const writeReportAndReturn = (result: LocalOcrResponse, extraNotes: string): LocalOcrResponse => {
    // Apply built-in English and reminder dataset spell correction for 90%+ accuracy
    if (result.text && result.text.trim()) {
      const lexiconRes = correctTextWithLexicon(result.text);
      if (lexiconRes.corrections.length > 0) {
        result.text = lexiconRes.correctedText;
        result.lines = result.lines.map((l) => {
          const lCorr = correctTextWithLexicon(l.text);
          return {
            ...l,
            text: lCorr.correctedText,
            confidence: Math.max(l.confidence, lCorr.accuracyScore / 100),
          };
        });
        appliedSteps.push(`lexicon-corrected(${lexiconRes.corrections.length})`);
      }
      result.accuracyScore = lexiconRes.accuracyScore;
      result.verifiedWords = lexiconRes.wordsVerified;
      result.totalWords = lexiconRes.totalWords;
    }

    if (runExportDir) {
      const linesReport = result.lines
        .map((l, idx) => `Line ${idx + 1} (${Math.round(l.confidence * 100)}%): ${l.text}`)
        .join('\n');
      const report = [
        `Profile: ${ocrProfile}`,
        `Detected style: ${result.detectedStyle} (pageScore=${detection.score})`,
        `Pipeline steps: ${appliedSteps.join(' -> ')}`,
        `Engine: ${result.recognizer} | Quality: ${result.qualityScore}`,
        extraNotes,
        linesReport ? `---\n${linesReport}` : '',
        `---`,
        result.text,
      ]
        .filter(Boolean)
        .join('\n');
      fs.writeFileSync(path.join(runExportDir, 'extracted_text.txt'), report, 'utf8');
    }
    return result;
  };

  // Explicit Printed Profile (kept deterministic for unit tests & direct API requests)
  if (ocrProfile === 'printed' && printedModelReady) {
    const printed = await runOfflinePrintedOnBoxes(regions);
    return writeReportAndReturn(
      {
        text: printed.text,
        lines: printed.lines,
        profile: ocrProfile,
        detectedStyle: 'printed',
        detection,
        recognizer: 'tesseract-host',
        qualityScore: printed.qualityScore,
        skewAngle,
        darkInverted,
        perspectiveWarped,
        bordersCropped,
        dpiScale,
        appliedSteps,
        regionsCount: regions.length,
        processedPreview,
        exportDir: runExportDir,
      },
      'Engine: Tesseract LSTM (Offline Printed)'
    );
  }

  // Explicit Handwriting Profile (kept deterministic for unit tests & direct API requests)
  if (ocrProfile === 'handwriting' && handwritingModelReady) {
    const lineBoxes = getOrderedLineBoxesForRegions(regions);
    const hw = await runOfflineHandwritingOnLines(lineBoxes, true);
    return writeReportAndReturn(
      {
        text: hw.text,
        lines: hw.lines,
        profile: ocrProfile,
        detectedStyle: 'handwriting',
        detection,
        recognizer: 'trocr-host',
        qualityScore: hw.qualityScore,
        skewAngle,
        darkInverted,
        perspectiveWarped,
        bordersCropped,
        dpiScale,
        appliedSteps,
        regionsCount: regions.length,
        processedPreview,
        droppedHallucinations: hw.droppedHallucinations,
        exportDir: runExportDir,
      },
      `Engine: TrOCR (Offline Handwriting) | Lines segmented: ${lineBoxes.length} | Kept: ${hw.lines.length} | Dropped: ${hw.droppedHallucinations}`
    );
  }

  // AUTO MODE: Multi-Region Mixed Printed + Handwriting Support
  if (ocrProfile === 'auto' && regions.length > 1 && printedModelReady && handwritingModelReady) {
    const mergedLines: LocalOcrLineResult[] = [];
    let usedPrinted = false;
    let usedHandwriting = false;
    let totalDropped = 0;

    for (const reg of regions) {
      const regCleanBin = cropRegionWithPadding(cleanBinary, w, h, reg, 8);
      const regDet = detectHandwriting(regCleanBin.data, regCleanBin.width, regCleanBin.height);
      const regSegBin = maskBinaryOutsideBox(closedForSeg, w, h, reg);
      const regLineBoxes = segmentLines(regSegBin, w, h);

      if (!regDet.isHandwriting) {
        const pr = await runOfflinePrintedOnBoxes([reg]);
        const prPlaus = scoreTextPlausibility(pr.text, pr.qualityScore);
        if (
          (!pr.text || pr.qualityScore < 0.7 || prPlaus < 0.56) &&
          regLineBoxes.length > 0
        ) {
          const hw = await runOfflineHandwritingOnLines(regLineBoxes, false);
          const hwPlaus = scoreTextPlausibility(hw.text, hw.qualityScore);
          totalDropped += hw.droppedHallucinations;
          if (hw.text && (hwPlaus > prPlaus + 0.05 || !pr.text)) {
            mergedLines.push(...hw.lines);
            usedHandwriting = true;
            continue;
          }
        }
        if (pr.text) {
          mergedLines.push(...pr.lines);
          usedPrinted = true;
        }
      } else {
        const hw = await runOfflineHandwritingOnLines(regLineBoxes, false);
        const hwPlaus = scoreTextPlausibility(hw.text, hw.qualityScore);
        totalDropped += hw.droppedHallucinations;
        if (!hw.text || hw.qualityScore < 0.65 || hwPlaus < 0.56) {
          const pr = await runOfflinePrintedOnBoxes([reg]);
          const prPlaus = scoreTextPlausibility(pr.text, pr.qualityScore);
          if (pr.text && (prPlaus >= hwPlaus + 0.08 || !hw.text)) {
            mergedLines.push(...pr.lines);
            usedPrinted = true;
            continue;
          }
        }
        if (hw.text) {
          mergedLines.push(...hw.lines);
          usedHandwriting = true;
        }
      }
    }

    const finalStyle: DetectedStyle =
      usedPrinted && usedHandwriting
        ? 'mixed'
        : usedHandwriting
          ? 'handwriting'
          : usedPrinted
            ? 'printed'
            : initialStyle;
    const finalRecognizer: RecognizerEngine =
      finalStyle === 'mixed'
        ? 'hybrid-host'
        : finalStyle === 'handwriting'
          ? 'trocr-host'
          : 'tesseract-host';

    if (finalStyle === 'mixed') {
      appliedSteps.push('auto-mixed-printed-handwriting');
    }

    const avgConf =
      mergedLines.length > 0
        ? mergedLines.reduce((acc, l) => acc + l.confidence, 0) / mergedLines.length
        : 0;

    return writeReportAndReturn(
      {
        text: mergedLines.map((l) => l.text).join('\n'),
        lines: mergedLines,
        profile: ocrProfile,
        detectedStyle: finalStyle,
        detection,
        recognizer: finalRecognizer,
        qualityScore: Number(avgConf.toFixed(2)),
        skewAngle,
        darkInverted,
        perspectiveWarped,
        bordersCropped,
        dpiScale,
        appliedSteps,
        regionsCount: regions.length,
        processedPreview,
        droppedHallucinations: totalDropped,
        exportDir: runExportDir,
      },
      `Engine: ${finalRecognizer} (Multi-Region Auto)`
    );
  }

  // AUTO MODE (Single-Region or Single-Engine Fallback) with Bidirectional Cross-Check + Line-Level Hybrid Recovery
  if (!detection.isHandwriting && printedModelReady) {
    const printed = await runOfflinePrintedOnBoxes(regions);
    const printedPlaus = scoreTextPlausibility(printed.text, printed.qualityScore);

    if (handwritingModelReady) {
      const lineBoxes = getOrderedLineBoxesForRegions(regions).filter(
        (b) => b.y1 - b.y0 + 1 >= 12
      );

      // Bidirectional cross-check: e.g. neat disconnected print handwriting that scored < 0.42 in detectHandwriting
      if (!printed.text || printed.qualityScore < 0.7 || printedPlaus < 0.58) {
        const hw = await runOfflineHandwritingOnLines(lineBoxes, true);
        const hwPlaus = scoreTextPlausibility(hw.text, hw.qualityScore);

        if (hw.text && (hwPlaus > printedPlaus + 0.05 || !printed.text)) {
          appliedSteps.push('auto-crosscheck-trocr');
          return writeReportAndReturn(
            {
              text: hw.text,
              lines: hw.lines,
              profile: ocrProfile,
              detectedStyle: 'handwriting',
              detection,
              recognizer: 'trocr-host',
              qualityScore: hw.qualityScore,
              skewAngle,
              darkInverted,
              perspectiveWarped,
              bordersCropped,
              dpiScale,
              appliedSteps,
              regionsCount: regions.length,
              processedPreview,
              droppedHallucinations: hw.droppedHallucinations,
              exportDir: runExportDir,
            },
            `Engine: TrOCR (Offline Bidirectional Cross-Check)`
          );
        }
      }

      // Single-region mixed lines recovery: strong printed lines + weak/missed handwritten lines in the same block
      const strongPrintedLines = printed.lines.filter((l) => l.confidence >= 0.62);
      const needsHybridLineCheck =
        strongPrintedLines.length > 0 &&
        (printed.lines.some((l) => l.confidence < 0.62) ||
          lineBoxes.length > printed.lines.length);

      if (needsHybridLineCheck) {
        const uncoveredOrWeakBoxes = lineBoxes.filter((box) => {
          const bestOverlap = strongPrintedLines.reduce(
            (best, pl) => Math.max(best, verticalOverlapRatio(box, pl.bbox)),
            0
          );
          return bestOverlap < 0.4;
        });

        if (uncoveredOrWeakBoxes.length > 0 && uncoveredOrWeakBoxes.length < lineBoxes.length) {
          const hwPatch = await runOfflineHandwritingOnLines(uncoveredOrWeakBoxes, false);
          if (hwPatch.lines.length > 0) {
            const combinedLines = [...strongPrintedLines, ...hwPatch.lines].sort((a, b) =>
              Math.abs(a.bbox.y0 - b.bbox.y0) > 16 ? a.bbox.y0 - b.bbox.y0 : a.bbox.x0 - b.bbox.x0
            );
            const avgConf =
              combinedLines.reduce((acc, l) => acc + l.confidence, 0) / combinedLines.length;
            appliedSteps.push('auto-hybrid-lines');
            return writeReportAndReturn(
              {
                text: combinedLines.map((l) => l.text).join('\n'),
                lines: combinedLines,
                profile: ocrProfile,
                detectedStyle: 'mixed',
                detection,
                recognizer: 'hybrid-host',
                qualityScore: Number(avgConf.toFixed(2)),
                skewAngle,
                darkInverted,
                perspectiveWarped,
                bordersCropped,
                dpiScale,
                appliedSteps,
                regionsCount: regions.length,
                processedPreview,
                droppedHallucinations: hwPatch.droppedHallucinations,
                exportDir: runExportDir,
              },
              `Engine: Hybrid Tesseract+TrOCR (Mixed Single-Region)`
            );
          }
        }
      }
    }

    return writeReportAndReturn(
      {
        text: printed.text,
        lines: printed.lines,
        profile: ocrProfile,
        detectedStyle: 'printed',
        detection,
        recognizer: 'tesseract-host',
        qualityScore: printed.qualityScore,
        skewAngle,
        darkInverted,
        perspectiveWarped,
        bordersCropped,
        dpiScale,
        appliedSteps,
        regionsCount: regions.length,
        processedPreview,
        exportDir: runExportDir,
      },
      'Engine: Tesseract LSTM (Offline Printed)'
    );
  }

  // Handwriting primary path (plus Printed cross-check when auto)
  const lineBoxes = getOrderedLineBoxesForRegions(regions);
  const hw = await runOfflineHandwritingOnLines(lineBoxes, true);
  const hwPlaus = scoreTextPlausibility(hw.text, hw.qualityScore);

  if (ocrProfile === 'auto' && printedModelReady && !detection.isHandwriting && (hw.qualityScore < 0.65 || !hw.text || hwPlaus < 0.56)) {
    const printedFallback = await runOfflinePrintedOnBoxes(regions);
    const printedPlaus = scoreTextPlausibility(printedFallback.text, printedFallback.qualityScore);
    if (printedFallback.text && (printedPlaus >= hwPlaus + 0.08 || !hw.text)) {
      appliedSteps.push('auto-crosscheck-tesseract');
      return writeReportAndReturn(
        {
          text: printedFallback.text,
          lines: printedFallback.lines,
          profile: ocrProfile,
          detectedStyle: 'printed',
          detection,
          recognizer: 'tesseract-host',
          qualityScore: printedFallback.qualityScore,
          skewAngle,
          darkInverted,
          perspectiveWarped,
          bordersCropped,
          dpiScale,
          appliedSteps,
          regionsCount: regions.length,
          processedPreview,
          droppedHallucinations: hw.droppedHallucinations,
          exportDir: runExportDir,
        },
        'Engine: Tesseract LSTM (Offline Printed Cross-Check)'
      );
    }
  }

  return writeReportAndReturn(
    {
      text: hw.text,
      lines: hw.lines,
      profile: ocrProfile,
      detectedStyle: 'handwriting',
      detection,
      recognizer: 'trocr-host',
      qualityScore: hw.qualityScore,
      skewAngle,
      darkInverted,
      perspectiveWarped,
      bordersCropped,
      dpiScale,
      appliedSteps,
      regionsCount: regions.length,
      processedPreview,
      droppedHallucinations: hw.droppedHallucinations,
      exportDir: runExportDir,
    },
    `Engine: TrOCR (Offline Handwriting) | Lines segmented: ${lineBoxes.length} | Kept: ${hw.lines.length} | Dropped: ${hw.droppedHallucinations}`
  );
}
