import {
  toGrayscale,
  autoInvertIfDarkMode,
  flattenBackground,
  applyMedianFilter3x3,
  enhanceInkContrast,
  binarizeSauvola,
  applyMorphologicalOp,
  twoPassContourVoting,
} from './preprocess';
import {
  correctDocumentPerspective,
  deskewGrayscale,
  deslantGrayscale,
} from './geometry';
import {
  cropDarkBordersAndMargins,
  removeRuledLines,
  filterTextComponents,
} from './handwriting/cleanup';
import { detectHandwriting } from './handwriting/detect';
import { segmentLines, splitWideLineBox, cropLine } from './handwriting/lines';
import { postprocessHandwrittenLines } from './postprocess';
import { correctTextWithLexicon } from './wordDataset';
import { parseReminderLocally } from '../../../server/localReminderParser';

export type BrowserOcrProfile = 'auto' | 'printed' | 'handwriting';

let trocrPipelinePromise: Promise<any> | null = null;

async function getBrowserTrOcrPipeline(onProgress?: (msg: string) => void) {
  if (!trocrPipelinePromise) {
    onProgress?.('Loading in-browser TrOCR handwriting model (cached locally after first load)...');
    trocrPipelinePromise = (async () => {
      const { pipeline, env } = await import('@huggingface/transformers');
      env.allowRemoteModels = true;
      env.allowLocalModels = false;
      return pipeline('image-to-text', 'Xenova/trocr-base-handwritten', {
        dtype: 'q8',
      });
    })().catch((err) => {
      trocrPipelinePromise = null;
      throw err;
    });
  }
  return trocrPipelinePromise;
}

async function decodeImageToGrayscale(dataUrl: string, maxWidth = 1400): Promise<{
  gray: Uint8ClampedArray;
  width: number;
  height: number;
}> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const natW = img.naturalWidth || img.width;
      const natH = img.naturalHeight || img.height;
      const scale = natW > maxWidth ? maxWidth / natW : 1;
      const w = Math.max(1, Math.round(natW * scale));
      const h = Math.max(1, Math.round(natH * scale));
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) {
        reject(new Error('Canvas 2D context unavailable'));
        return;
      }
      ctx.drawImage(img, 0, 0, w, h);
      const imgData = ctx.getImageData(0, 0, w, h);
      resolve({
        gray: toGrayscale(imgData.data, w, h, 4),
        width: w,
        height: h,
      });
    };
    img.onerror = () => reject(new Error('Failed to decode image in browser'));
    img.src = dataUrl;
  });
}

function grayscaleToDataUrl(gray: Uint8ClampedArray, width: number, height: number): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';
  const imgData = ctx.createImageData(width, height);
  for (let i = 0; i < gray.length; i++) {
    const v = gray[i];
    const j = i * 4;
    imgData.data[j] = v;
    imgData.data[j + 1] = v;
    imgData.data[j + 2] = v;
    imgData.data[j + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL('image/png');
}

async function recognizePrintedInBrowser(
  gray: Uint8ClampedArray,
  width: number,
  height: number
): Promise<{
  text: string;
  lines: Array<{ text: string; confidence: number; bbox: { x0: number; y0: number; x1: number; y1: number } }>;
}> {
  const { createWorker } = await import('tesseract.js');
  const worker = await createWorker('eng');
  try {
    const pngDataUrl = grayscaleToDataUrl(gray, width, height);
    const { data } = await worker.recognize(pngDataUrl);
    const lines = (data.text || '')
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter(Boolean)
      .map((t) => ({
        text: t,
        confidence: Number(Math.max(0.5, Math.min(0.95, (data.confidence || 70) / 100)).toFixed(2)),
        bbox: { x0: 0, y0: 0, x1: width, y1: height },
      }));
    return {
      text: lines.map((l) => l.text).join('\n'),
      lines,
    };
  } finally {
    await worker.terminate();
  }
}

/**
 * Pure client-side OCR fallback used when Snap2 is hosted statically (e.g. GitHub Pages)
 * without a Node/Express backend. Mirrors the 4-stage preprocessing + TrOCR/Tesseract pipeline.
 */
export async function runBrowserLocalOcr(params: {
  image: string;
  ocrProfile: BrowserOcrProfile;
  prompt?: string;
  onProgress?: (msg: string) => void;
}) {
  params.onProgress?.('Preprocessing image in browser...');
  const decoded = await decodeImageToGrayscale(params.image);
  let workGray = decoded.gray;
  let w = decoded.width;
  let h = decoded.height;

  const appliedSteps: string[] = ['browser-grayscale'];
  const { data: polar, inverted } = autoInvertIfDarkMode(workGray, w, h);
  workGray = polar;
  if (inverted) appliedSteps.push('color-inversion');

  const persp = correctDocumentPerspective(workGray, w, h);
  if (persp.warped) {
    workGray = persp.data;
    w = persp.width;
    h = persp.height;
    appliedSteps.push('perspective-dewarp');
  } else {
    const cropped = cropDarkBordersAndMargins(workGray, w, h);
    if (cropped.cropped) {
      workGray = cropped.data;
      w = cropped.width;
      h = cropped.height;
      appliedSteps.push('border-crop');
    }
  }

  workGray = flattenBackground(workGray, w, h);
  workGray = applyMedianFilter3x3(workGray, w, h);
  appliedSteps.push('illumination-flatten', 'median-3x3');

  const radius = Math.max(14, Math.min(36, Math.round(Math.min(w, h) / 28)));
  const rawBin = binarizeSauvola(workGray, w, h, radius, 0.16);
  const deskew = deskewGrayscale(workGray, rawBin, w, h, { minCorrectionDeg: 0.4, maxAngleDeg: 28 });
  if (deskew.corrected) {
    workGray = deskew.data;
    appliedSteps.push(`deskew(${deskew.angleDeg}deg)`);
  }

  const noRules = removeRuledLines(workGray, w, h);
  const contrastGray = enhanceInkContrast(noRules, w, h);
  const contourVote = twoPassContourVoting(noRules, w, h);
  const cleanBinary = w >= 280 && h >= 200 ? filterTextComponents(contourVote.binary, w, h) : contourVote.binary;
  const closedForSeg = applyMorphologicalOp(cleanBinary, w, h, 'close', 1);
  const lineBoxes = segmentLines(closedForSeg, w, h, { skipFilter: true });
  const detection = detectHandwriting(cleanBinary, w, h, { lines: lineBoxes, skipFilter: true });

  const processedPreview = grayscaleToDataUrl(contrastGray, w, h);
  const useHandwriting =
    params.ocrProfile === 'handwriting' ||
    (params.ocrProfile === 'auto' && detection.isHandwriting);

  let text = '';
  let lines: Array<{ text: string; confidence: number; bbox: { x0: number; y0: number; x1: number; y1: number } }> = [];
  let recognizer: 'trocr-host' | 'tesseract-host' = useHandwriting ? 'trocr-host' : 'tesseract-host';
  let detectedStyle: 'handwriting' | 'printed' = useHandwriting ? 'handwriting' : 'printed';

  if (useHandwriting && lineBoxes.length > 0) {
    try {
      const rec = await getBrowserTrOcrPipeline(params.onProgress);
      params.onProgress?.(`Recognizing ${lineBoxes.length} handwritten lines in browser...`);
      const rawResults: Array<{ text: string; confidence: number; bbox: any }> = [];

      for (const box of lineBoxes) {
        const chunks = splitWideLineBox(closedForSeg, w, h, box, 6.5);
        const chunkTexts: string[] = [];
        for (const chunkBox of chunks) {
          const cropped = cropLine(contrastGray, w, h, chunkBox, 16);
          const deslanted = deslantGrayscale(cropped.data, cropped.width, cropped.height);
          const lineUrl = grayscaleToDataUrl(deslanted.data, deslanted.width, deslanted.height);
          if (!lineUrl) continue;
          const out = await rec(lineUrl, {
            num_beams: 4,
            max_new_tokens: 64,
            early_stopping: true,
          });
          const first = Array.isArray(out) ? out[0] : out;
          if (typeof first?.generated_text === 'string' && first.generated_text.trim()) {
            chunkTexts.push(first.generated_text.trim());
          }
        }
        const joined = chunkTexts.join(' ').trim();
        if (joined) {
          rawResults.push({ text: joined, confidence: 0.88, bbox: box });
        }
      }

      const cleaned = postprocessHandwrittenLines(rawResults);
      lines = cleaned.map((l) => ({
        text: l.text,
        confidence: l.confidence,
        bbox: l.bbox || { x0: 0, y0: 0, x1: w, y1: h },
      }));
      text = lines.map((l) => l.text).join('\n');
    } catch {
      params.onProgress?.('Falling back to in-browser Tesseract OCR...');
      recognizer = 'tesseract-host';
      const printed = await recognizePrintedInBrowser(contrastGray, w, h);
      text = printed.text;
      lines = printed.lines;
    }
  } else {
    params.onProgress?.('Running in-browser Tesseract OCR...');
    const printed = await recognizePrintedInBrowser(contrastGray, w, h);
    text = printed.text;
    lines = printed.lines;
  }

  if (text.trim()) {
    const lex = correctTextWithLexicon(text);
    text = lex.correctedText;
    lines = lines.map((l) => ({
      ...l,
      text: correctTextWithLexicon(l.text).correctedText,
    }));
  }

  const deterministicData = text.trim()
    ? parseReminderLocally(
        text,
        params.prompt,
        detectedStyle === 'handwriting' ? ['handwritten'] : ['ocr']
      )
    : undefined;

  return {
    success: true,
    text,
    lines,
    detectedStyle,
    recognizer,
    skewAngle: deskew.angleDeg,
    appliedSteps,
    regionsCount: 1,
    processedPreview,
    data: deterministicData,
  };
}
