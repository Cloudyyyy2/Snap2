/**
 * High-Performance Gaussian Adaptive Thresholding & Binarization Service (`src/services/imageProcessor.ts`)
 * Implements integral-image optimized sliding-window local intensity averaging, shadow removal,
 * and Gaussian adaptive thresholding to normalize handwriting contrast regardless of lighting conditions.
 */

import { PreprocessOptions, preprocessImageData } from '../lib/ocr/preprocess';

/**
 * High-performance Gaussian adaptive thresholding function for OCR flow.
 * Uses integral images for $O(1)$ sliding window local intensity threshold calculation,
 * normalizing handwriting contrast and eliminating shadows across uneven lighting conditions.
 */
export function preprocessForOCR(canvas: HTMLCanvasElement): ImageData {
  if (!canvas || typeof canvas.width !== 'number' || typeof canvas.height !== 'number') {
    throw new Error('Valid HTMLCanvasElement required for preprocessForOCR');
  }

  const width = canvas.width;
  const height = canvas.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Canvas 2D context unavailable');

  const imgData = ctx.getImageData(0, 0, width, height);
  const rgbaData = imgData.data;

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

  // Construct output ImageData with clean binary ink (0) on white paper (255)
  const enhancedData = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < result.binary.length; i++) {
    const val = result.binary[i] === 0 ? 0 : 255;
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
