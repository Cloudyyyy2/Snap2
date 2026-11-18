import {
  rgbaToGrayscale,
  binarizeSauvola,
  segmentWords,
  segmentLines,
  WordBox,
  LineBox,
  cropToSquare,
} from './handwriting/cleanup';

export interface NormalizedBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectedWordSquare {
  id: string;
  wordIndex: number;
  lineIndex: number;
  box: LineBox;
  squareBox: { x0: number; y0: number; x1: number; y1: number; size: number };
  // Normalized (0..1) percentage coordinates for responsive UI overlays
  normalized: {
    word: NormalizedBox;
    square: NormalizedBox;
  };
  thumbnailUrl?: string;
}

export interface WordDetectionResult {
  width: number;
  height: number;
  words: DetectedWordSquare[];
  lines: LineBox[];
  totalInkCoverage: number;
  // Bounding square enclosing all detected text with balanced padding
  documentSquareBox: {
    x0: number;
    y0: number;
    x1: number;
    y1: number;
    size: number;
    normalized: NormalizedBox;
  } | null;
}

/**
 * Loads any image data URL or image source into an HTMLImageElement safely.
 */
export function loadImageElement(source: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(new Error('Failed to load image for on-device word detection.'));
    img.src = source;
  });
}

/**
 * Runs pure client-side on-device word-edge detection and square-box fitting.
 * Works completely offline in the browser without network calls.
 */
export async function detectWordsOnImage(
  source: string | HTMLCanvasElement,
  options: {
    maxDimension?: number;
    generateThumbnails?: boolean;
    maxThumbnails?: number;
  } = {}
): Promise<WordDetectionResult> {
  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D | null;

  const maxDim = options.maxDimension ?? 960;

  if (typeof source === 'string') {
    const img = await loadImageElement(source);
    let origW = img.naturalWidth || img.width;
    let origH = img.naturalHeight || img.height;
    if (origW === 0 || origH === 0) {
      return { width: 0, height: 0, words: [], lines: [], totalInkCoverage: 0, documentSquareBox: null };
    }

    let scale = 1;
    if (Math.max(origW, origH) > maxDim) {
      scale = maxDim / Math.max(origW, origH);
    }
    const w = Math.max(40, Math.round(origW * scale));
    const h = Math.max(40, Math.round(origH * scale));

    canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('Canvas 2D context unavailable');
    ctx.drawImage(img, 0, 0, w, h);
  } else {
    canvas = source;
    ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('Canvas 2D context unavailable');
  }

  const w = canvas.width;
  const h = canvas.height;
  const imgData = ctx.getImageData(0, 0, w, h);
  const gray = rgbaToGrayscale(imgData.data, w, h);

  // Resolution-adaptive Sauvola binarization radius
  const sauvolaRadius = Math.max(12, Math.min(32, Math.round(Math.min(w, h) / 32)));
  const binary = binarizeSauvola(gray, w, h, sauvolaRadius, 0.18);

  const lines = segmentLines(binary, w, h);
  const rawWords = segmentWords(binary, w, h, lines);

  let inkPixels = 0;
  for (let i = 0; i < binary.length; i++) {
    if (binary[i] === 0) inkPixels++;
  }
  const totalInkCoverage = inkPixels / (w * h);

  // Compute union bounding box enclosing all detected words
  let docMinX = w;
  let docMaxX = -1;
  let docMinY = h;
  let docMaxY = -1;

  for (const word of rawWords) {
    if (word.x0 < docMinX) docMinX = word.x0;
    if (word.x1 > docMaxX) docMaxX = word.x1;
    if (word.y0 < docMinY) docMinY = word.y0;
    if (word.y1 > docMaxY) docMaxY = word.y1;
  }

  let documentSquareBox: WordDetectionResult['documentSquareBox'] = null;
  if (docMaxX >= docMinX && docMaxY >= docMinY) {
    const textW = docMaxX - docMinX + 1;
    const textH = docMaxY - docMinY + 1;
    const pad = Math.round(Math.max(12, Math.max(textW, textH) * 0.12));
    const side = Math.max(textW, textH) + pad * 2;
    const cx = (docMinX + docMaxX) / 2;
    const cy = (docMinY + docMaxY) / 2;
    const sqX0 = Math.round(cx - side / 2);
    const sqY0 = Math.round(cy - side / 2);

    documentSquareBox = {
      x0: sqX0,
      y0: sqY0,
      x1: sqX0 + side - 1,
      y1: sqY0 + side - 1,
      size: side,
      normalized: {
        x: sqX0 / w,
        y: sqY0 / h,
        width: side / w,
        height: side / h,
      },
    };
  }

  const thumbLimit = options.maxThumbnails ?? 24;
  const generateThumbs = options.generateThumbnails !== false;

  const detectedWords: DetectedWordSquare[] = rawWords.map((word, idx) => {
    const wordW = word.x1 - word.x0 + 1;
    const wordH = word.y1 - word.y0 + 1;

    let thumbnailUrl: string | undefined;
    if (generateThumbs && idx < thumbLimit) {
      try {
        const sq = cropToSquare(gray, w, h, word, 128, 0.2);
        const thumbCanvas = document.createElement('canvas');
        thumbCanvas.width = sq.size;
        thumbCanvas.height = sq.size;
        const thumbCtx = thumbCanvas.getContext('2d');
        if (thumbCtx) {
          const thumbImgData = thumbCtx.createImageData(sq.size, sq.size);
          for (let i = 0; i < sq.data.length; i++) {
            const v = sq.data[i];
            const j = i * 4;
            thumbImgData.data[j] = v;
            thumbImgData.data[j + 1] = v;
            thumbImgData.data[j + 2] = v;
            thumbImgData.data[j + 3] = 255;
          }
          thumbCtx.putImageData(thumbImgData, 0, 0);
          thumbnailUrl = thumbCanvas.toDataURL('image/jpeg', 0.88);
        }
      } catch {
        // ignore thumb generation failure
      }
    }

    return {
      id: `word-${word.lineIndex}-${word.wordIndex}-${idx}`,
      wordIndex: word.wordIndex,
      lineIndex: word.lineIndex,
      box: {
        x0: word.x0,
        y0: word.y0,
        x1: word.x1,
        y1: word.y1,
      },
      squareBox: word.squareBox,
      normalized: {
        word: {
          x: word.x0 / w,
          y: word.y0 / h,
          width: wordW / w,
          height: wordH / h,
        },
        square: {
          x: word.squareBox.x0 / w,
          y: word.squareBox.y0 / h,
          width: word.squareBox.size / w,
          height: word.squareBox.size / h,
        },
      },
      thumbnailUrl,
    };
  });

  return {
    width: w,
    height: h,
    words: detectedWords,
    lines,
    totalInkCoverage,
    documentSquareBox,
  };
}

/**
 * Crops a specific word or region into a pure 1:1 square Data URL right on the local device,
 * with clean paper padding and optional target resolution (e.g. 384x384 or original square size).
 */
export async function cropImageToSquareDataUrl(
  imageSource: string,
  box: LineBox,
  targetSquareSize = 384,
  padRatio = 0.2
): Promise<string> {
  const img = await loadImageElement(imageSource);
  const origW = img.naturalWidth || img.width;
  const origH = img.naturalHeight || img.height;

  const boxW = Math.max(1, box.x1 - box.x0 + 1);
  const boxH = Math.max(1, box.y1 - box.y0 + 1);
  const pad = Math.round(Math.max(8, Math.max(boxW, boxH) * padRatio));
  const side = Math.max(boxW, boxH) + pad * 2;

  const cx = (box.x0 + box.x1) / 2;
  const cy = (box.y0 + box.y1) / 2;
  const sqX0 = Math.round(cx - side / 2);
  const sqY0 = Math.round(cy - side / 2);

  const canvas = document.createElement('canvas');
  canvas.width = targetSquareSize;
  canvas.height = targetSquareSize;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D context unavailable');

  // Fill with clean white paper
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, targetSquareSize, targetSquareSize);

  // Compute destination placement: map [sqX0, sqY0, side, side] -> [0, 0, targetSquareSize, targetSquareSize]
  const scale = targetSquareSize / side;
  const dstX = (0 - sqX0) * scale;
  const dstY = (0 - sqY0) * scale;
  const dstW = origW * scale;
  const dstH = origH * scale;

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, dstX, dstY, dstW, dstH);

  return canvas.toDataURL('image/jpeg', 0.92);
}
