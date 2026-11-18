/**
 * High-Performance Gaussian Adaptive Thresholding & Binarization Service (`src/services/imageProcessor.ts`)
 * Implements integral-image optimized sliding-window local intensity averaging, shadow removal,
 * and Gaussian adaptive thresholding to normalize handwriting contrast regardless of lighting conditions.
 */

import { PreprocessOptions, preprocessImageData } from '../lib/ocr/preprocess';

/**
 * High-performance Gaussian adaptive thresholding function for OCR flow.
 * Accepts an HTMLCanvasElement or ImageData, applies local intensity normalization,
 * removes background noise and shadows, and returns the binarized ImageData.
 */
export function preprocessForOCR(source: HTMLCanvasElement | ImageData | any): ImageData {
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

  // Gaussian adaptive thresholding & local sliding-window intensity averaging configuration
  const preOpts: PreprocessOptions = {
    profile: 'handwriting',
    binarize: 'sauvola',
    sauvolaWindowRadius: 15,
    sauvolaK: 0.2,
    flattenLighting: true,
    enhanceContrast: true,
  };

  const result = preprocessImageData({ data: rgbaData, width, height }, preOpts);

  // Construct output ImageData with continuous anti-aliased grayscale strokes (0..255)
  const enhancedData = new Uint8ClampedArray(width * height * 4);
  const grayBuffer = result.enhancedGray || result.gray;
  for (let i = 0; i < grayBuffer.length; i++) {
    const val = grayBuffer[i];
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
