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
  computeRecommendedDpiScale,
} from './handwriting/cleanup';
import { detectHandwriting } from './handwriting/detect';
import { segmentLines, splitWideLineBox, cropLine, LineBox } from './handwriting/lines';
import { postprocessHandwrittenLines } from './postprocess';
import { hasErraticTesseractArtifacts, scoreTextPlausibility } from './plausibility';
import { correctTextWithLexicon } from './wordDataset';
import { parseReminderLocally } from '../../../server/localReminderParser';

export type BrowserOcrProfile = 'auto' | 'printed' | 'handwriting';

let trocrPipelinePromise: Promise<any> | null = null;

function isMobileBrowser(): boolean {
  if (typeof navigator === 'undefined') return false;
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent || ''
  );
}

async function getBrowserTrOcrPipeline(onProgress?: (msg: string) => void) {
  if (!trocrPipelinePromise) {
    const mobile = isMobileBrowser();
    const modelId = mobile
      ? 'Xenova/trocr-small-handwritten'
      : 'Xenova/trocr-base-handwritten';
    onProgress?.(
      mobile
        ? 'Loading compact in-browser TrOCR model (mobile-optimized)...'
        : 'Loading in-browser TrOCR handwriting model (cached locally after first load)...'
    );
    trocrPipelinePromise = (async () => {
      const { pipeline, env } = await import('@huggingface/transformers');
      env.allowRemoteModels = true;
      env.allowLocalModels = false;
      // Single-threaded, non-proxied WASM avoids COOP/COEP & worker memory issues on GitHub Pages / mobile
      if (env.backends?.onnx?.wasm) {
        env.backends.onnx.wasm.numThreads = 1;
        env.backends.onnx.wasm.proxy = false;
      }
      return pipeline('image-to-text', modelId, {
        dtype: 'q8',
      });
    })().catch((err) => {
      trocrPipelinePromise = null;
      throw err;
    });
  }
  return trocrPipelinePromise;
}

async function decodeImageToGrayscale(dataUrl: string, maxWidth = 960): Promise<{
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

function upscaleGrayscaleCanvas(
  gray: Uint8ClampedArray,
  width: number,
  height: number,
  targetW: number,
  targetH: number
): Uint8ClampedArray {
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = width;
  srcCanvas.height = height;
  const srcCtx = srcCanvas.getContext('2d');
  if (!srcCtx) return gray;
  const srcImg = srcCtx.createImageData(width, height);
  for (let i = 0; i < gray.length; i++) {
    const v = gray[i];
    const j = i * 4;
    srcImg.data[j] = v;
    srcImg.data[j + 1] = v;
    srcImg.data[j + 2] = v;
    srcImg.data[j + 3] = 255;
  }
  srcCtx.putImageData(srcImg, 0, 0);

  const dstCanvas = document.createElement('canvas');
  dstCanvas.width = targetW;
  dstCanvas.height = targetH;
  const dstCtx = dstCanvas.getContext('2d', { willReadFrequently: true });
  if (!dstCtx) return gray;
  dstCtx.imageSmoothingEnabled = true;
  dstCtx.imageSmoothingQuality = 'high';
  dstCtx.drawImage(srcCanvas, 0, 0, targetW, targetH);
  const dstImg = dstCtx.getImageData(0, 0, targetW, targetH);
  return toGrayscale(dstImg.data, targetW, targetH, 4);
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

  let noRules = removeRuledLines(workGray, w, h);
  const dpiRadius = Math.max(14, Math.min(36, Math.round(Math.min(w, h) / 28)));
  const dpiBin = binarizeSauvola(noRules, w, h, dpiRadius, 0.16);
  const dpiCleanBin = w >= 280 && h >= 200 ? filterTextComponents(dpiBin, w, h) : dpiBin;
  const rawDpiScale = computeRecommendedDpiScale(dpiCleanBin, w, h);
  const maxDim = isMobileBrowser() ? 960 : 1280;
  const effectiveScale = rawDpiScale >= 1.5 ? Math.min(rawDpiScale, maxDim / Math.max(w, h)) : 1;
  if (effectiveScale >= 1.25) {
    const targetW = Math.max(1, Math.round(w * effectiveScale));
    const targetH = Math.max(1, Math.round(h * effectiveScale));
    noRules = upscaleGrayscaleCanvas(noRules, w, h, targetW, targetH);
    w = targetW;
    h = targetH;
    appliedSteps.push(`dpi-upscale(${Number(effectiveScale.toFixed(2))}x)`);
  }

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

  const runBrowserTrOcr = async (targetBoxes: LineBox[]) => {
    const rec = await getBrowserTrOcrPipeline(params.onProgress);
    params.onProgress?.(`Recognizing ${targetBoxes.length} handwritten lines in browser...`);
    const rawResults: Array<{ text: string; confidence: number; bbox: LineBox }> = [];

    for (let lineIdx = 0; lineIdx < targetBoxes.length; lineIdx++) {
      const box = targetBoxes[lineIdx];
      params.onProgress?.(`Recognizing line ${lineIdx + 1} of ${targetBoxes.length}...`);
      const chunks = splitWideLineBox(closedForSeg, w, h, box, 6.5);
      const chunkTexts: string[] = [];
      for (const chunkBox of chunks) {
        const cropped = cropLine(contrastGray, w, h, chunkBox, 16);
        const deslanted = deslantGrayscale(cropped.data, cropped.width, cropped.height);
        const lineUrl = grayscaleToDataUrl(deslanted.data, deslanted.width, deslanted.height);
        if (!lineUrl) continue;
        const out = await rec(lineUrl, {
          num_beams: isMobileBrowser() ? 2 : 4,
          max_new_tokens: 64,
          early_stopping: true,
        });
        const first = Array.isArray(out) ? out[0] : out;
        if (typeof first?.generated_text === 'string' && first.generated_text.trim()) {
          chunkTexts.push(first.generated_text.trim());
        }
        await new Promise((r) => setTimeout(r, 0));
      }
      const joined = chunkTexts.join(' ').trim();
      if (joined) {
        rawResults.push({ text: joined, confidence: 0.88, bbox: box });
      }
    }

    const cleaned = postprocessHandwrittenLines(rawResults);
    const hwLines = cleaned.map((l) => ({
      text: l.text,
      confidence: l.confidence,
      bbox: l.bbox || { x0: 0, y0: 0, x1: w, y1: h },
    }));
    const hwText = hwLines.map((l) => l.text).join('\n');
    const qualityScore =
      hwLines.length > 0
        ? hwLines.reduce((acc, l) => acc + l.confidence, 0) / hwLines.length
        : 0;
    return { text: hwText, lines: hwLines, qualityScore };
  };

  let text = '';
  let lines: Array<{ text: string; confidence: number; bbox: { x0: number; y0: number; x1: number; y1: number } }> = [];
  let recognizer: 'trocr-host' | 'tesseract-host' = useHandwriting ? 'trocr-host' : 'tesseract-host';
  let detectedStyle: 'handwriting' | 'printed' = useHandwriting ? 'handwriting' : 'printed';

  if (useHandwriting && lineBoxes.length > 0) {
    try {
      const hw = await runBrowserTrOcr(lineBoxes);
      lines = hw.lines;
      text = hw.text;
    } catch (trocrErr: any) {
      const reason = trocrErr?.message || String(trocrErr);
      throw new Error(
        `In-browser handwriting model failed to load (${reason}). Please check your internet connection for first-time model download or use Printed mode.`
      );
    }
  } else {
    params.onProgress?.('Running in-browser Tesseract OCR...');
    const printed = await recognizePrintedInBrowser(contrastGray, w, h);
    text = printed.text;
    lines = printed.lines;

    if (params.ocrProfile === 'auto' && lineBoxes.length > 0) {
      const printedQuality =
        printed.lines.length > 0
          ? printed.lines.reduce((acc, l) => acc + l.confidence, 0) / printed.lines.length
          : 0;
      const printedPlaus = scoreTextPlausibility(printed.text, printedQuality, printed.lines);
      const printedErratic = hasErraticTesseractArtifacts(printed.text, printed.lines);

      const shouldCrossCheck =
        !printed.text.trim() ||
        printedErratic ||
        printedPlaus < 0.68 ||
        printedQuality < 0.78 ||
        lineBoxes.length >= 3;

      if (shouldCrossCheck) {
        try {
          const hw = await runBrowserTrOcr(lineBoxes);
          const hwPlaus = scoreTextPlausibility(hw.text, hw.qualityScore, hw.lines);
          if (
            hw.text.trim() &&
            (hwPlaus > printedPlaus + 0.02 ||
              printedErratic ||
              printedQuality < 0.72 ||
              !printed.text.trim())
          ) {
            text = hw.text;
            lines = hw.lines;
            recognizer = 'trocr-host';
            detectedStyle = 'handwriting';
            appliedSteps.push('auto-crosscheck-trocr');
          }
        } catch (trocrErr: any) {
          if (!printed.text.trim() || printedErratic || printedQuality < 0.72) {
            const reason = trocrErr?.message || String(trocrErr);
            throw new Error(
              `In-browser handwriting model failed to load (${reason}). Please check your internet connection for first-time model download or switch to Vision AI.`
            );
          }
        }
      }
    }
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
    engineOrigin: 'browser' as const,
    skewAngle: deskew.angleDeg,
    appliedSteps,
    regionsCount: 1,
    processedPreview,
    data: deterministicData,
  };
}
