import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { createWorker, PSM, Worker } from 'tesseract.js';

const TESSDATA_DIR = path.resolve(process.cwd(), 'public/models/tessdata');
const ENG_TRAINEDDATA_GZ = path.join(TESSDATA_DIR, 'eng.traineddata.gz');
const ENG_TRAINEDDATA = path.join(TESSDATA_DIR, 'eng.traineddata');

let workerPromise: Promise<Worker> | null = null;

export function isPrintedModelAvailable(customDir = TESSDATA_DIR): boolean {
  const gz = path.join(customDir, 'eng.traineddata.gz');
  const raw = path.join(customDir, 'eng.traineddata');
  return (
    (fs.existsSync(raw) && fs.statSync(raw).size > 100_000) ||
    (fs.existsSync(gz) && fs.statSync(gz).size > 100_000)
  );
}

export function getActivePrintedModelName(
  customDir = TESSDATA_DIR
): 'tessdata_full' | 'tessdata_fast' | 'missing' {
  if (!isPrintedModelAvailable(customDir)) return 'missing';
  const marker = path.join(customDir, '.tessdata-full');
  const raw = path.join(customDir, 'eng.traineddata');
  if (fs.existsSync(marker) || (fs.existsSync(raw) && fs.statSync(raw).size > 20_000_000)) {
    return 'tessdata_full';
  }
  return 'tessdata_fast';
}

export async function getPrintedWorker(): Promise<Worker> {
  if (!isPrintedModelAvailable()) {
    throw new Error(
      `Offline Tesseract data missing at ${ENG_TRAINEDDATA} or ${ENG_TRAINEDDATA_GZ}. Run "npm run models:handwriting" first.`
    );
  }

  if (!workerPromise) {
    // Prefer uncompressed eng.traineddata directly so tessdata_best loads without gunzipping each start
    const hasRaw = fs.existsSync(ENG_TRAINEDDATA) && fs.statSync(ENG_TRAINEDDATA).size > 100_000;
    workerPromise = createWorker('eng', 1, {
      langPath: TESSDATA_DIR,
      cachePath: TESSDATA_DIR,
      gzip: !hasRaw,
    }).catch((err) => {
      workerPromise = null;
      throw err;
    });
  }

  return workerPromise;
}

export interface PrintedOcrLineResult {
  text: string;
  confidence: number;
  bbox: { x0: number; y0: number; x1: number; y1: number };
}

/**
 * Runs offline Tesseract LSTM OCR on a preprocessed (upscaled, median-denoised,
 * Sauvola/Otsu cleaned) grayscale/binary PNG buffer.
 */
export async function recognizePrintedBuffer(params: {
  grayOrBinary: Uint8ClampedArray;
  width: number;
  height: number;
  psm?: PSM;
}): Promise<{ text: string; confidence: number; lines: PrintedOcrLineResult[] }> {
  const worker = await getPrintedWorker();
  await worker.setParameters({
    tessedit_pageseg_mode: params.psm ?? PSM.AUTO,
    user_defined_dpi: '300',
    preserve_interword_spaces: '1',
  });

  const pngBuffer = await sharp(Buffer.from(params.grayOrBinary), {
    raw: { width: params.width, height: params.height, channels: 1 },
  })
    .png()
    .toBuffer();

  const { data } = await worker.recognize(pngBuffer, {}, { blocks: true });
  const rawText = (data.text || '')
    .split(/\r?\n/)
    .map((line) => line.replace(/[ \t]+/g, ' ').trim())
    .filter((line) => line.length > 0 && !/^[^A-Za-z0-9]{1,3}$/.test(line))
    .join('\n');

  const extractedLines = (data.blocks ?? []).flatMap((b) =>
    (b.paragraphs ?? []).flatMap((p) => p.lines ?? [])
  );

  const blocksLines = extractedLines
    .map((l) => ({
      text: (l.text || '').replace(/[ \t]+/g, ' ').trim(),
      confidence: Number(((l.confidence || 0) / 100).toFixed(2)),
      bbox: l.bbox || { x0: 0, y0: 0, x1: params.width, y1: params.height },
    }))
    .filter((l) => l.text.length > 0 && !/^[^A-Za-z0-9]{1,3}$/.test(l.text));

  const lines: PrintedOcrLineResult[] =
    blocksLines.length > 0
      ? blocksLines
      : rawText
          .split('\n')
          .filter(Boolean)
          .map((t) => ({
            text: t,
            confidence: Number(((data.confidence || 0) / 100).toFixed(2)),
            bbox: { x0: 0, y0: 0, x1: params.width, y1: params.height },
          }));

  const normalizedConf = Number(Math.max(0, Math.min(1, (data.confidence || 0) / 100)).toFixed(2));

  return {
    text: lines.map((l) => l.text).join('\n') || rawText,
    confidence: normalizedConf,
    lines,
  };
}
