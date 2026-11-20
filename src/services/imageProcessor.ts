/**
 * Grayscale-Preserving OCR Normalization Service (`src/services/imageProcessor.ts`)
 * Implements integral-image background illumination flattening and local contrast stretching
 * to normalize handwriting and printed contrast while preserving continuous anti-aliased strokes (0..255).
 * Optionally supports binary Sauvola output when `{ mode: 'binary' }` is explicitly requested.
 */

import { PreprocessOptions, preprocessImageData } from '../lib/ocr/preprocess';

export interface PreprocessForOcrOptions {
  mode?: 'grayscale' | 'binary';
}

/**
 * Normalizes an HTMLCanvasElement or ImageData for OCR / Vision AI submission.
 * Flattens uneven shadows and enhances ink contrast while preserving continuous anti-aliased
 * grayscale strokes by default (`mode = 'grayscale'`), or returns Sauvola binary when `mode = 'binary'`.
 */
export function preprocessForOCR(
  source: HTMLCanvasElement | ImageData | any,
  options: PreprocessForOcrOptions = {}
): ImageData {
  if (!source || typeof source.width !== 'number' || typeof source.height !== 'number') {
    throw new Error('Valid HTMLCanvasElement or ImageData required for preprocessForOCR');
  }

  const width = source.width;
  const height = source.height;
  let rgbaData: Uint8ClampedArray;

  if (source && typeof source.getContext === 'function') {
    const ctx = source.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('Canvas 2D context unavailable');
    const imgData = ctx.getImageData(0, 0, width, height);
    rgbaData = imgData.data instanceof Uint8ClampedArray ? imgData.data : new Uint8ClampedArray(imgData.data);
  } else if (source && source.data) {
    rgbaData = source.data instanceof Uint8ClampedArray ? source.data : new Uint8ClampedArray(source.data);
  } else if (typeof document !== 'undefined') {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('Canvas 2D context unavailable');
    ctx.drawImage(source as CanvasImageSource, 0, 0);
    rgbaData = ctx.getImageData(0, 0, width, height).data;
  } else {
    throw new Error('Unsupported source type passed to preprocessForOCR');
  }

  // Background illumination flattening & contrast normalization (with optional Sauvola binary)
  const preOpts: PreprocessOptions = {
    profile: 'handwriting',
    binarize: 'sauvola',
    sauvolaWindowRadius: 15,
    sauvolaK: 0.2,
    flattenLighting: true,
    enhanceContrast: true,
  };

  const result = preprocessImageData({ data: rgbaData, width, height }, preOpts);

  // Construct output ImageData with continuous anti-aliased grayscale strokes (0..255) by default
  const enhancedData = new Uint8ClampedArray(width * height * 4);
  const outputBuffer =
    options.mode === 'binary' ? result.binary : result.enhancedGray || result.gray;
  for (let i = 0; i < outputBuffer.length; i++) {
    const val = outputBuffer[i];
    const idx = i * 4;
    enhancedData[idx] = val;
    enhancedData[idx + 1] = val;
    enhancedData[idx + 2] = val;
    enhancedData[idx + 3] = 255;
  }

  if (typeof ImageData !== 'undefined') {
    return new ImageData(enhancedData, width, height);
  }

  return {
    data: enhancedData,
    width,
    height,
  } as unknown as ImageData;
}
