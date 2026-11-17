import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { pipeline, env, RawImage } from '@huggingface/transformers';
import { flattenBackground, removeRuledLines, binarizeSauvola } from '../src/lib/ocr/handwriting/cleanup';
import { segmentLines, cropLine, CroppedLine } from '../src/lib/ocr/handwriting/lines';
import { detectHandwriting, HandwritingDetectionResult } from '../src/lib/ocr/handwriting/detect';
import { postprocessHandwrittenLines } from '../src/lib/ocr/postprocess';

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
  recognizer: 'trocr-host' | 'vision-fallback';
  exportDir?: string;
}

const MODEL_DIR_NAME = 'trocr-small-handwritten';
const MODELS_ROOT = path.resolve(process.cwd(), 'public/models');
const MODEL_DIR = path.join(MODELS_ROOT, MODEL_DIR_NAME);

const REQUIRED_MODEL_FILES = [
  'config.json',
  'preprocessor_config.json',
  'tokenizer.json',
  'onnx/encoder_model_quantized.onnx',
  'onnx/decoder_model_merged_quantized.onnx',
];

let recognizerPromise: Promise<any> | null = null;

export function getModelDir(): string {
  return MODEL_DIR;
}

export function isHandwritingModelAvailable(customModelDir = MODEL_DIR): boolean {
  return REQUIRED_MODEL_FILES.every((rel) => {
    const full = path.join(customModelDir, rel);
    return fs.existsSync(full) && fs.statSync(full).size > 0;
  });
}

export async function getHandwritingRecognizer() {
  if (!isHandwritingModelAvailable()) {
    throw new Error(
      `Local handwriting model missing at ${MODEL_DIR}. Run "npm run models:handwriting" first.`
    );
  }

  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.localModelPath = MODELS_ROOT;

  if (!recognizerPromise) {
    recognizerPromise = pipeline('image-to-text', MODEL_DIR_NAME, {
      dtype: 'q8',
    }).catch((err) => {
      recognizerPromise = null;
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

/**
 * Runs TrOCR-small-handwritten sequentially line-by-line to keep CPU usage bounded.
 */
export async function recognizeLines(
  lineCrops: CroppedLine[]
): Promise<Array<{ text: string; confidence: number }>> {
  const rec = await getHandwritingRecognizer();
  const results: Array<{ text: string; confidence: number }> = [];

  for (const crop of lineCrops) {
    const rawImg = grayscaleToRgbRawImage(crop.data, crop.width, crop.height);
    const out = await rec(rawImg, { max_new_tokens: 64 });
    const first = Array.isArray(out) ? out[0] : out;
    const text = typeof first?.generated_text === 'string' ? first.generated_text.trim() : '';

    // Estimate line confidence heuristically when token logprobs are unavailable
    const confidence =
      typeof first?.score === 'number'
        ? first.score
        : text.length > 0 && !/^[?\W]+$/.test(text)
          ? 0.78
          : 0.35;

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

export function decodeBase64Image(imageInput: string): Buffer {
  const matches = imageInput.match(/^data:(.+);base64,(.+)$/);
  const base64 = matches ? matches[2] : imageInput;
  return Buffer.from(base64, 'base64');
}

/**
 * Full shared preprocessing + handwriting detection + TrOCR recognition pipeline.
 * Also writes stage PNGs and extracted_text.txt into ocr-exports/run-<ts>/ when exportStages is enabled.
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

  // Decode & auto-orient upright grayscale, capped at 1800px wide for responsive CPU execution
  const { data: rawGrayBuffer, info } = await sharp(params.imageBuffer)
    .rotate()
    .resize({ width: 1800, withoutEnlargement: true })
    .grayscale()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const w = info.width;
  const h = info.height;
  const rawGray = new Uint8ClampedArray(rawGrayBuffer);

  // Shared cleanup
  const flattened = flattenBackground(rawGray, w, h);
  const noRules = removeRuledLines(flattened, w, h);
  const binary = binarizeSauvola(noRules, w, h);
  const detection = detectHandwriting(binary, w, h);

  const detectedStyle: 'printed' | 'handwriting' = detection.isHandwriting ? 'handwriting' : 'printed';
  const useHandwritingPath =
    ocrProfile === 'handwriting' || (ocrProfile === 'auto' && detectedStyle === 'handwriting');

  const modelReady = isHandwritingModelAvailable();

  let runExportDir: string | undefined;
  if (exportStages) {
    runExportDir = path.join(exportRoot, `run-${Date.now()}`);
    fs.mkdirSync(runExportDir, { recursive: true });
    await saveGrayscalePng(path.join(runExportDir, '03b_background_flattened.png'), flattened, w, h);
    await saveGrayscalePng(path.join(runExportDir, '03c_ruled_lines_removed.png'), noRules, w, h);
  }

  if (!modelReady || !useHandwritingPath) {
    if (runExportDir) {
      const report = [
        `Profile: ${ocrProfile}`,
        `Detected style: ${detectedStyle} (score=${detection.score})`,
        `Engine: Vision fallback (modelReady=${modelReady})`,
      ].join('\n');
      fs.writeFileSync(path.join(runExportDir, 'extracted_text.txt'), report, 'utf8');
    }

    return {
      text: '',
      lines: [],
      profile: ocrProfile,
      detectedStyle,
      detection,
      recognizer: 'vision-fallback',
      exportDir: runExportDir,
    };
  }

  const lineBoxes = segmentLines(binary, w, h);
  const lineCrops = lineBoxes.map((box) => cropLine(noRules, w, h, box, 12));

  if (runExportDir) {
    for (let i = 0; i < lineCrops.length; i++) {
      const idx = String(i + 1).padStart(2, '0');
      await saveGrayscalePng(
        path.join(runExportDir, `06_line_${idx}.png`),
        lineCrops[i].data,
        lineCrops[i].width,
        lineCrops[i].height
      );
    }
  }

  const recognized = await recognizeLines(lineCrops);
  const rawLines = recognized.map((r, i) => ({
    text: r.text,
    confidence: r.confidence,
    bbox: lineBoxes[i],
  }));

  const cleanedLines = postprocessHandwrittenLines(rawLines);
  const finalLines: LocalOcrLineResult[] = cleanedLines.map((l) => ({
    text: l.text,
    confidence: l.confidence,
    bbox: l.bbox || { x0: 0, y0: 0, x1: w, y1: h },
  }));

  const fullText = finalLines.map((l) => l.text).join('\n');

  if (runExportDir) {
    const linesReport = finalLines
      .map((l, idx) => `Line ${idx + 1} (${Math.round(l.confidence * 100)}%): ${l.text}`)
      .join('\n');
    const report = [
      `Profile: ${ocrProfile}`,
      `Detected style: ${detectedStyle} (score=${detection.score})`,
      `Engine: TrOCR (trocr-small-handwritten)`,
      `Lines segmented: ${lineCrops.length}`,
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
    exportDir: runExportDir,
  };
}
