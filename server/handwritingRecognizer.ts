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
import { isPrintedModelAvailable, recognizePrintedBuffer } from './printedRecognizer';

export type OcrProfile = 'auto' | 'printed' | 'handwriting';

export interface LocalOcrLineResult {
  text: string;
  confidence: number;
  bbox: { x0: number; y0: number; x1: number; y1: number };
}

export interface LocalOcrResponse {
  text: string;
  lines: LocalOcrLineResult[];
  profile: OcrProfile;
  detectedStyle: 'printed' | 'handwriting';
  detection: HandwritingDetectionResult;
  recognizer: 'trocr-host' | 'tesseract-host' | 'vision-fallback';
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

/**
 * Runs TrOCR-small-handwritten sequentially line-by-line (chunking ultra-wide lines
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

/**
 * Full 4-Stage Offline Image Processing + Dual Recognition Pipeline:
 * 1. Luminance & Contrast Normalization (grayscale collapse, polarity inversion, Retinex flattening, Otsu/Sauvola)
 * 2. Geometric & Spatial Correction (perspective quad dewarping, deskewing, 300+ DPI scaling)
 * 3. Noise Removal & Edge Preservation (3x3 median, bilateral smoothing, border cropping, morphological ops)
 * 4. Layout & Region Segmentation (XY-cut text regions, ruled-line removal, line/word segmentation)
 * -> Dual Offline Recognizers (Tesseract LSTM for printed, TrOCR for handwriting)
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

  // Stage 2a & 3c: Perspective Quad Dewarping & Dark Border Cropping BEFORE illumination flattening
  // so dark desk surfaces retain contrast against the paper sheet rather than normalizing to 255
  let workGray = polarNormalized;
  let perspectiveWarped = false;
  let bordersCropped = false;
  const persp = correctDocumentPerspective(workGray, w, h);
  if (persp.warped) {
    workGray = persp.data;
    w = persp.width;
    h = persp.height;
    perspectiveWarped = true;
    appliedSteps.push('perspective-dewarp');
  }

  const croppedDoc = cropDarkBordersAndMargins(workGray, w, h);
  if (croppedDoc.cropped) {
    workGray = croppedDoc.data;
    w = croppedDoc.width;
    h = croppedDoc.height;
    bordersCropped = true;
    appliedSteps.push('border-crop');
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

  // Stage 4b (Pass 1) + Stage 2b: Ruled-Line Cleanup & Projection Deskewing
  let noRules = removeRuledLines(workGray, w, h);
  let binary = binarizeSauvola(noRules, w, h);
  let cleanBinary = w >= 280 && h >= 200 ? filterTextComponents(binary, w, h) : binary;

  const deskew = deskewGrayscale(workGray, cleanBinary, w, h, { minCorrectionDeg: 0.6 });
  const skewAngle = deskew.angleDeg;
  if (deskew.corrected) {
    workGray = deskew.data;
    noRules = removeRuledLines(deskew.data, w, h);
    binary = binarizeSauvola(noRules, w, h);
    cleanBinary = w >= 280 && h >= 200 ? filterTextComponents(binary, w, h) : binary;
    appliedSteps.push(`deskew(${skewAngle}deg)`);
  }
  appliedSteps.push('ruled-lines-removed');

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
    binary = binarizeSauvola(noRules, w, h);
    cleanBinary = w >= 280 && h >= 200 ? filterTextComponents(binary, w, h) : binary;
    appliedSteps.push(`dpi-upscale(${dpiScale}x)`);
  }

  // Stage 1b + 3d: Combined Sauvola/Otsu Adaptive Threshold + Morphological Stroke Bridging
  const otsuBin = binarizeOtsu(noRules, w, h);
  // Blend Otsu cleanly only to keep strong uniform printed ink, while Sauvola handles uneven shadows
  const blendedBinary = new Uint8ClampedArray(w * h);
  for (let i = 0; i < blendedBinary.length; i++) {
    blendedBinary[i] = cleanBinary[i] === 0 || (otsuBin[i] === 0 && noRules[i] < 135) ? 0 : 255;
  }
  const regionBinary =
    w >= 280 && h >= 200 ? filterTextComponents(blendedBinary, w, h) : blendedBinary;

  // Morphological closing bridges broken faint pencil/pen strokes for line segmentation
  const closedForSeg = applyMorphologicalOp(regionBinary, w, h, 'close', 1);
  appliedSteps.push('sauvola-otsu-binarize', 'morph-close');

  const contrastGray = enhanceInkContrast(noRules, w, h);
  appliedSteps.push('ink-contrast-stretch');

  // Stage 4: Layout Region Segmentation & Style Detection
  // Note: detectHandwriting uses un-closed cleanBinary so morph-close never bridges adjacent printed glyphs into fake cursive!
  const regions = segmentLayoutRegions(regionBinary, w, h);
  appliedSteps.push(`layout-regions(${regions.length})`);

  const detection = detectHandwriting(cleanBinary, w, h);
  const detectedStyle: 'printed' | 'handwriting' = detection.isHandwriting
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

  const runOfflinePrinted = async (): Promise<{
    text: string;
    lines: LocalOcrLineResult[];
    qualityScore: number;
  }> => {
    const activeRegions =
      regions.length > 0 ? regions : [{ x0: 0, y0: 0, x1: w - 1, y1: h - 1, inkPixels: 100 }];

    const allLines: LocalOcrLineResult[] = [];
    const regionTexts: string[] = [];

    for (const reg of activeRegions) {
      const crop = cropRegionWithPadding(contrastGray, w, h, reg, 18);
      const res = await recognizePrintedBuffer({
        grayOrBinary: crop.data,
        width: crop.width,
        height: crop.height,
        psm: activeRegions.length > 1 ? PSM.SINGLE_BLOCK : PSM.AUTO,
      });
      if (res.text.trim()) {
        regionTexts.push(res.text.trim());
        for (const l of res.lines) {
          allLines.push({
            text: l.text,
            confidence: l.confidence,
            bbox: {
              x0: reg.x0 + l.bbox.x0,
              y0: reg.y0 + l.bbox.y0,
              x1: reg.x0 + l.bbox.x1,
              y1: reg.y0 + l.bbox.y1,
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

  const wantPrinted =
    ocrProfile === 'printed' || (ocrProfile === 'auto' && detectedStyle === 'printed');

  // Path A: Offline Printed OCR (Tesseract LSTM)
  if (wantPrinted && printedModelReady) {
    const printed = await runOfflinePrinted();

    if (runExportDir) {
      const report = [
        `Profile: ${ocrProfile}`,
        `Detected style: ${detectedStyle} (score=${detection.score})`,
        `Pipeline steps: ${appliedSteps.join(' -> ')}`,
        `Engine: Tesseract LSTM (Offline Printed) | Quality: ${printed.qualityScore}`,
        `---`,
        printed.text,
      ].join('\n');
      fs.writeFileSync(path.join(runExportDir, 'extracted_text.txt'), report, 'utf8');
    }

    return {
      text: printed.text,
      lines: printed.lines,
      profile: ocrProfile,
      detectedStyle,
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
    };
  }

  // Fallback if neither local model exists
  if (!handwritingModelReady && !printedModelReady) {
    return {
      text: '',
      lines: [],
      profile: ocrProfile,
      detectedStyle,
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

  // Path B: Offline Handwriting OCR (TrOCR on layout-ordered regions & morphological closed lines)
  const hasSideBySideColumns =
    regions.length > 1 &&
    regions.some((a, idx) =>
      regions.some(
        (b, jdx) =>
          idx !== jdx &&
          (a.x1 < b.x0 - 20 || b.x1 < a.x0 - 20) &&
          Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0) > 20
      )
    );
  const orderedLineBoxes: LineBox[] = [];
  if (hasSideBySideColumns) {
    for (const reg of regions) {
      const regBin = new Uint8ClampedArray(w * h).fill(255);
      for (let y = reg.y0; y <= reg.y1; y++) {
        const rowOff = y * w;
        for (let x = reg.x0; x <= reg.x1; x++) {
          regBin[rowOff + x] = closedForSeg[rowOff + x];
        }
      }
      orderedLineBoxes.push(...segmentLines(regBin, w, h));
    }
  }
  const lineBoxes =
    orderedLineBoxes.length > 0 ? orderedLineBoxes : segmentLines(closedForSeg, w, h);

  const chunkPlans = lineBoxes.map((box) => ({
    lineBox: box,
    chunks: splitWideLineBox(closedForSeg, w, h, box, 5.0),
  }));

  const flatChunks: CroppedLine[] = [];
  for (const plan of chunkPlans) {
    for (const chunkBox of plan.chunks) {
      flatChunks.push(cropLine(contrastGray, w, h, chunkBox, 16));
    }
  }

  if (runExportDir) {
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

  const recognizedChunks = handwritingModelReady ? await recognizeLines(flatChunks) : [];
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

  // Offline Cross-Check in Auto mode: if TrOCR confidence is weak and Tesseract is ready, compare offline
  if (ocrProfile === 'auto' && printedModelReady && (qualityScore < 0.65 || !fullText)) {
    const printedFallback = await runOfflinePrinted();
    if (printedFallback.text && printedFallback.qualityScore >= Math.max(0.6, qualityScore + 0.08)) {
      if (runExportDir) {
        const report = [
          `Profile: ${ocrProfile}`,
          `Detected style: ${detectedStyle} -> offline cross-check promoted Tesseract`,
          `Pipeline steps: ${appliedSteps.join(' -> ')}`,
          `Engine: Tesseract LSTM (Offline Printed Cross-Check) | Quality: ${printedFallback.qualityScore}`,
          `---`,
          printedFallback.text,
        ].join('\n');
        fs.writeFileSync(path.join(runExportDir, 'extracted_text.txt'), report, 'utf8');
      }

      return {
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
        droppedHallucinations,
        exportDir: runExportDir,
      };
    }
  }

  if (runExportDir) {
    const linesReport = finalLines
      .map((l, idx) => `Line ${idx + 1} (${Math.round(l.confidence * 100)}%): ${l.text}`)
      .join('\n');
    const report = [
      `Profile: ${ocrProfile}`,
      `Detected style: ${detectedStyle} (score=${detection.score})`,
      `Pipeline steps: ${appliedSteps.join(' -> ')}`,
      `Engine: TrOCR (Offline Handwriting) | Quality: ${qualityScore}`,
      `Lines segmented: ${lineBoxes.length} | Kept: ${finalLines.length} | Dropped: ${droppedHallucinations}`,
      `---`,
      linesReport,
      `---`,
      fullText,
    ].join('\n');
    fs.writeFileSync(path.join(runExportDir, 'extracted_text.txt'), report, 'utf8');
  }

  return {
    text: fullText,
    lines: finalLines,
    profile: ocrProfile,
    detectedStyle,
    detection,
    recognizer: 'trocr-host',
    qualityScore,
    skewAngle,
    darkInverted,
    perspectiveWarped,
    bordersCropped,
    dpiScale,
    appliedSteps,
    regionsCount: regions.length,
    processedPreview,
    droppedHallucinations,
    exportDir: runExportDir,
  };
}
