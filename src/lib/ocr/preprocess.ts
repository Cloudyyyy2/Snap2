/**
 * Pure TypeScript OCR Image Preprocessing Utility
 *
 * Works identically on browser ImageData (RGBA/RGB) and Node.js raw pixel buffers
 * with zero native or platform-specific dependencies:
 * 1. Grayscale Conversion (ITU-R BT.601 luminance + alpha compositing onto white paper)
 * 2. Noise Reduction (3x3 Median filtering, separable Gaussian blur, 5x5 Bilateral smoothing)
 * 3. Luminance & Polarity Normalization (dark-mode inversion, Retinex background flattening, contrast stretch)
 * 4. Adaptive Binarization & Morphology (Sauvola local thresholding, Otsu global thresholding, morph ops)
 * 5. Optional Stage-2 Geometry integration (perspective dewarp & projection deskew)
 */

import { correctDocumentPerspective } from './geometry';

export type ChannelCount = 1 | 3 | 4;

export type DenoiseMethod =
  | 'auto'
  | 'none'
  | 'median'
  | 'gaussian'
  | 'bilateral'
  | 'median+gaussian'
  | 'median+bilateral';

export type BinarizeMethod = 'sauvola' | 'otsu' | 'hybrid';

export interface PreprocessOptions {
  /** Input channel count (1 = grayscale, 3 = RGB, 4 = RGBA). Auto-detected if omitted. */
  channels?: ChannelCount;
  /** OCR profile tuning ('auto' | 'printed' | 'handwriting'). Default: 'auto' */
  profile?: 'auto' | 'printed' | 'handwriting';
  /** Invert light-on-dark images (e.g. dark-mode UI) into dark ink on white paper. Default: true */
  autoInvert?: boolean;
  /** Normalize uneven lighting & shadows via local background estimation. Default: true */
  flattenLighting?: boolean;
  /** Noise reduction strategy. Default: 'median+gaussian' */
  denoise?: DenoiseMethod;
  /** Gaussian blur standard deviation when Gaussian noise reduction is used. Default: 0.85 */
  gaussianSigma?: number;
  /** Explicit Gaussian kernel radius. Default: auto-derived from gaussianSigma */
  gaussianRadius?: number;
  /** Thresholding strategy for the binary mask. Default: 'sauvola' */
  binarize?: BinarizeMethod;
  /** Half-window radius for Sauvola local adaptive thresholding. Default: 15 */
  sauvolaWindowRadius?: number;
  /** Sauvola sensitivity factor k (typically 0.15 - 0.35). Default: 0.2 */
  sauvolaK?: number;
  /** Stretch dark ink contrast & snap bright paper regions cleanly to 255. Default: true */
  enhanceContrast?: boolean;
  /** Optional morphological cleanup on the binary mask ('none' | 'open' | 'close'). Default: 'none' */
  morphology?: 'none' | 'open' | 'close';
  /** Automatically rectify keystone/trapezoid document photos before cleanup. Default: false */
  perspective?: boolean;
  /** Automatically deskew tilted text lines on the binary/grayscale buffers. Default: false */
  deskew?: boolean;
}

export interface PreprocessResult {
  /** Luminance-normalized & denoised grayscale buffer (0 = ink, 255 = paper) */
  gray: Uint8ClampedArray;
  /** Contrast-enhanced grayscale buffer optimized for TrOCR / Tesseract recognition */
  enhancedGray: Uint8ClampedArray;
  /** Binary foreground/background mask (0 = ink, 255 = paper) */
  binary: Uint8ClampedArray;
  width: number;
  height: number;
  inverted: boolean;
  perspectiveWarped: boolean;
  skewAngle: number;
  appliedSteps: string[];
}

export interface ImageDataLike {
  data: Uint8ClampedArray | Uint8Array;
  width: number;
  height: number;
}

function buildIntegralImage(data: Uint8ClampedArray, w: number, h: number): Float64Array {
  const stride = w + 1;
  const integral = new Float64Array(stride * (h + 1));

  for (let y = 0; y < h; y++) {
    let rowSum = 0;
    const rowOffset = y * w;
    const prevIntRow = y * stride;
    const curIntRow = (y + 1) * stride;
    for (let x = 0; x < w; x++) {
      rowSum += data[rowOffset + x];
      integral[curIntRow + x + 1] = integral[prevIntRow + x + 1] + rowSum;
    }
  }

  return integral;
}

function buildSquaredIntegralImage(data: Uint8ClampedArray, w: number, h: number): Float64Array {
  const stride = w + 1;
  const integral = new Float64Array(stride * (h + 1));

  for (let y = 0; y < h; y++) {
    let rowSum = 0;
    const rowOffset = y * w;
    const prevIntRow = y * stride;
    const curIntRow = (y + 1) * stride;
    for (let x = 0; x < w; x++) {
      const v = data[rowOffset + x];
      rowSum += v * v;
      integral[curIntRow + x + 1] = integral[prevIntRow + x + 1] + rowSum;
    }
  }

  return integral;
}

function boxSum(
  integral: Float64Array,
  w: number,
  h: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number
): { sum: number; count: number } {
  const cx0 = Math.max(0, x0);
  const cy0 = Math.max(0, y0);
  const cx1 = Math.min(w - 1, x1);
  const cy1 = Math.min(h - 1, y1);
  const stride = w + 1;

  const sum =
    integral[(cy1 + 1) * stride + (cx1 + 1)] -
    integral[cy0 * stride + (cx1 + 1)] -
    integral[(cy1 + 1) * stride + cx0] +
    integral[cy0 * stride + cx0];

  const count = Math.max(1, (cx1 - cx0 + 1) * (cy1 - cy0 + 1));
  return { sum, count };
}

/**
 * Converts an RGBA (4-channel) buffer to 1-channel ITU-R BT.601 grayscale luminance,
 * compositing translucent pixels cleanly onto white paper (255).
 */
export function rgbaToGrayscale(
  rgba: Uint8ClampedArray | Uint8Array,
  w: number,
  h: number
): Uint8ClampedArray {
  const count = w * h;
  const out = new Uint8ClampedArray(count);
  if (count === 0) return out;

  for (let i = 0; i < count; i++) {
    const j = i * 4;
    const r = rgba[j];
    const g = rgba[j + 1];
    const b = rgba[j + 2];
    const a = rgba[j + 3];
    const luma = (77 * r + 150 * g + 29 * b + 128) >> 8;

    if (a >= 252) {
      out[i] = luma;
    } else if (a === 0) {
      out[i] = 255;
    } else {
      const alpha = a / 255;
      out[i] = Math.round(luma * alpha + 255 * (1 - alpha));
    }
  }

  return out;
}

/**
 * Converts an RGB (3-channel) buffer to 1-channel ITU-R BT.601 grayscale luminance.
 */
export function rgbToGrayscale(
  rgb: Uint8ClampedArray | Uint8Array,
  w: number,
  h: number
): Uint8ClampedArray {
  const count = w * h;
  const out = new Uint8ClampedArray(count);
  if (count === 0) return out;

  for (let i = 0; i < count; i++) {
    const j = i * 3;
    out[i] = (77 * rgb[j] + 150 * rgb[j + 1] + 29 * rgb[j + 2] + 128) >> 8;
  }

  return out;
}

/**
 * Converts a 1-channel, 3-channel RGB, or 4-channel RGBA buffer into a 1-channel
 * grayscale Uint8ClampedArray (auto-detecting channels when omitted).
 */
export function toGrayscale(
  input: Uint8ClampedArray | Uint8Array,
  w: number,
  h: number,
  channels?: ChannelCount
): Uint8ClampedArray {
  const pixelCount = w * h;
  if (pixelCount === 0) return new Uint8ClampedArray(0);

  let detectedChannels = channels;
  if (!detectedChannels) {
    if (input.length >= pixelCount * 4) detectedChannels = 4;
    else if (input.length >= pixelCount * 3) detectedChannels = 3;
    else detectedChannels = 1;
  }

  if (detectedChannels === 4) {
    return rgbaToGrayscale(input, w, h);
  }
  if (detectedChannels === 3) {
    return rgbToGrayscale(input, w, h);
  }
  return new Uint8ClampedArray(input.subarray(0, pixelCount));
}

/**
 * Converts a 1-channel grayscale buffer into an RGBA buffer (useful for Canvas ImageData preview).
 */
export function grayscaleToRgba(
  gray: Uint8ClampedArray,
  w: number,
  h: number
): Uint8ClampedArray {
  const count = Math.min(gray.length, w * h);
  const rgba = new Uint8ClampedArray(count * 4);
  for (let i = 0; i < count; i++) {
    const v = gray[i];
    const j = i * 4;
    rgba[j] = v;
    rgba[j + 1] = v;
    rgba[j + 2] = v;
    rgba[j + 3] = 255;
  }
  return rgba;
}

/**
 * Detects predominantly dark-background / light-foreground images (e.g. dark-mode
 * screenshots, terminal captures, chalkboards) and inverts them so text is dark ink
 * on bright paper for consistent downstream OCR preprocessing.
 */
export function autoInvertIfDarkMode(
  gray: Uint8ClampedArray,
  w: number,
  h: number
): { data: Uint8ClampedArray; inverted: boolean } {
  if (w === 0 || h === 0) {
    return { data: new Uint8ClampedArray(gray), inverted: false };
  }

  const samples: number[] = [];
  const step = Math.max(1, Math.floor((w * h) / 4096));
  let darkCount = 0;
  let brightCount = 0;

  for (let i = 0; i < gray.length; i += step) {
    const v = gray[i];
    samples.push(v);
    if (v < 85) darkCount++;
    if (v > 170) brightCount++;
  }

  if (samples.length === 0) {
    return { data: new Uint8ClampedArray(gray), inverted: false };
  }

  samples.sort((a, b) => a - b);
  const median = samples[Math.floor(samples.length * 0.5)];
  const p98 = samples[Math.min(samples.length - 1, Math.floor(samples.length * 0.98))];
  const darkFraction = darkCount / samples.length;
  const brightFraction = brightCount / samples.length;

  const isDarkMode =
    median < 92 &&
    p98 > 140 &&
    darkFraction >= 0.6 &&
    brightFraction >= 0.004 &&
    brightFraction <= 0.3;

  if (!isDarkMode) {
    return { data: new Uint8ClampedArray(gray), inverted: false };
  }

  const inverted = new Uint8ClampedArray(gray.length);
  for (let i = 0; i < gray.length; i++) {
    inverted[i] = 255 - gray[i];
  }
  return { data: inverted, inverted: true };
}

/**
 * Estimates local paper brightness with a large box blur (radius ~ 1/20 of min dimension)
 * and normalizes: out = clamp(255 * gray / background).
 * Removes camera shadows and uneven lighting while keeping ink strokes crisp on white paper.
 */
export function flattenBackground(
  gray: Uint8ClampedArray,
  w: number,
  h: number
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h);
  if (w === 0 || h === 0) return out;

  const radius = Math.max(18, Math.round(Math.min(w, h) / 12));
  const integral = buildIntegralImage(gray, w, h);

  for (let y = 0; y < h; y++) {
    const y0 = y - radius;
    const y1 = y + radius;
    const rowOffset = y * w;
    for (let x = 0; x < w; x++) {
      const { sum, count } = boxSum(integral, w, h, x - radius, y0, x + radius, y1);
      const bg = Math.max(12, sum / count);
      const normalized = Math.round((255 * gray[rowOffset + x]) / bg);
      out[rowOffset + x] = normalized > 255 ? 255 : normalized < 0 ? 0 : normalized;
    }
  }

  return out;
}

/**
 * Deepens foreground ink strokes while locking bright paper regions to clean white,
 * improving contrast for faint pencil or ballpoint pen writing without sharpening paper grain.
 */
export function enhanceInkContrast(
  gray: Uint8ClampedArray,
  w: number,
  h: number
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h);
  if (w === 0 || h === 0) return out;

  const samples: number[] = [];
  const step = Math.max(1, Math.floor((w * h) / 4096));
  for (let i = 0; i < gray.length; i += step) {
    samples.push(gray[i]);
  }
  samples.sort((a, b) => a - b);
  const paperWhite = Math.max(200, samples[Math.floor(samples.length * 0.85)] ?? 245);
  const whiteCutoff = Math.max(185, paperWhite - 16);
  const inkFloor = 22;
  const span = Math.max(40, whiteCutoff - inkFloor);

  for (let i = 0; i < gray.length; i++) {
    const v = gray[i];
    if (v >= whiteCutoff) {
      out[i] = 255;
    } else if (v <= inkFloor) {
      out[i] = 0;
    } else {
      const norm = (v - inkFloor) / span;
      const curved = Math.pow(norm, 2.2);
      out[i] = Math.max(0, Math.min(255, Math.round(curved * 240)));
    }
  }

  return out;
}

/**
 * 3x3 median filter that eliminates salt-and-pepper scanner/camera specks while
 * keeping character corners and strokes sharp without Gaussian blur.
 */
export function applyMedianFilter3x3(
  gray: Uint8ClampedArray,
  w: number,
  h: number
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(gray);
  if (w < 3 || h < 3) return out;

  const win = new Uint8Array(9);
  for (let y = 1; y < h - 1; y++) {
    const r0 = (y - 1) * w;
    const r1 = y * w;
    const r2 = (y + 1) * w;
    for (let x = 1; x < w - 1; x++) {
      win[0] = gray[r0 + x - 1];
      win[1] = gray[r0 + x];
      win[2] = gray[r0 + x + 1];
      win[3] = gray[r1 + x - 1];
      win[4] = gray[r1 + x];
      win[5] = gray[r1 + x + 1];
      win[6] = gray[r2 + x - 1];
      win[7] = gray[r2 + x];
      win[8] = gray[r2 + x + 1];
      win.sort();
      out[r1 + x] = win[4];
    }
  }
  return out;
}

/**
 * Separable 2-pass Gaussian smoothing filter (horizontal -> vertical) with clamped edge padding.
 * Reduces high-frequency camera sensor noise and JPEG compression mosquito artifacts before thresholding.
 */
export function applyGaussianBlur(
  gray: Uint8ClampedArray,
  w: number,
  h: number,
  sigma = 0.85,
  radius?: number
): Uint8ClampedArray {
  if (w < 2 || h < 2 || sigma <= 0.05) {
    return new Uint8ClampedArray(gray);
  }

  const r = Math.max(1, Math.min(6, radius ?? Math.ceil(sigma * 2)));
  const kernelSize = r * 2 + 1;
  const kernel = new Float32Array(kernelSize);
  const invTwoSigmaSq = -0.5 / (sigma * sigma);
  let kSum = 0;

  for (let i = -r; i <= r; i++) {
    const wVal = Math.exp(i * i * invTwoSigmaSq);
    kernel[i + r] = wVal;
    kSum += wVal;
  }
  for (let i = 0; i < kernelSize; i++) {
    kernel[i] /= kSum;
  }

  const tmp = new Float32Array(w * h);
  const out = new Uint8ClampedArray(w * h);

  // Horizontal pass
  for (let y = 0; y < h; y++) {
    const rowOff = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -r; k <= r; k++) {
        const sx = x + k < 0 ? 0 : x + k >= w ? w - 1 : x + k;
        acc += gray[rowOff + sx] * kernel[k + r];
      }
      tmp[rowOff + x] = acc;
    }
  }

  // Vertical pass
  for (let y = 0; y < h; y++) {
    const rowOff = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -r; k <= r; k++) {
        const sy = y + k < 0 ? 0 : y + k >= h ? h - 1 : y + k;
        acc += tmp[sy * w + x] * kernel[k + r];
      }
      const rounded = Math.round(acc);
      out[rowOff + x] = rounded < 0 ? 0 : rounded > 255 ? 255 : rounded;
    }
  }

  return out;
}

/**
 * Estimates high-frequency camera/scanner background noise using the 60th percentile
 * discrete Laplacian response (immune to crisp text edges which occupy < 25% of pixels).
 * Clean digital screenshots score ~0; noisy mobile camera/JPEG photos score >= 4.
 */
export function estimateNoiseLevel(
  gray: Uint8ClampedArray,
  w: number,
  h: number
): number {
  if (w < 16 || h < 16) return 0;

  const residuals: number[] = [];
  const stepX = Math.max(2, Math.floor(w / 48));
  const stepY = Math.max(2, Math.floor(h / 48));

  for (let y = 2; y < h - 2; y += stepY) {
    const r0 = (y - 1) * w;
    const r1 = y * w;
    const r2 = (y + 1) * w;
    for (let x = 2; x < w - 2; x += stepX) {
      const lap = Math.abs(
        4 * gray[r1 + x] - gray[r1 + x - 1] - gray[r1 + x + 1] - gray[r0 + x] - gray[r2 + x]
      );
      residuals.push(lap * 0.25);
    }
  }

  if (residuals.length === 0) return 0;
  residuals.sort((a, b) => a - b);
  return Number((residuals[Math.floor(residuals.length * 0.6)] ?? 0).toFixed(2));
}

/**
 * Edge-preserving bilateral filter (5x5 window) that suppresses camera sensor grain
 * and paper fiber texture while preserving sharp intensity transitions at ink edges.
 */
export function applyBilateralSmoothing(
  gray: Uint8ClampedArray,
  w: number,
  h: number,
  sigmaSpace = 2.0,
  sigmaRange = 24.0
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(gray);
  if (w < 5 || h < 5) return out;

  const radius = 2;
  const invTwoSpaceSq = -0.5 / (sigmaSpace * sigmaSpace);
  const invTwoRangeSq = -0.5 / (sigmaRange * sigmaRange);

  const spaceWeights = new Float32Array(25);
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      spaceWeights[(dy + radius) * 5 + (dx + radius)] = Math.exp(
        (dx * dx + dy * dy) * invTwoSpaceSq
      );
    }
  }

  const rangeLut = new Float32Array(256);
  for (let d = 0; d < 256; d++) {
    rangeLut[d] = Math.exp(d * d * invTwoRangeSq);
  }

  for (let y = radius; y < h - radius; y++) {
    const rowOff = y * w;
    for (let x = radius; x < w - radius; x++) {
      const center = gray[rowOff + x];
      let sumW = 0;
      let sumVal = 0;

      for (let dy = -radius; dy <= radius; dy++) {
        const sRow = (y + dy) * w;
        const wRow = (dy + radius) * 5;
        for (let dx = -radius; dx <= radius; dx++) {
          const val = gray[sRow + x + dx];
          const diff = Math.abs(val - center);
          const weight = spaceWeights[wRow + dx + radius] * rangeLut[diff];
          sumW += weight;
          sumVal += val * weight;
        }
      }

      out[rowOff + x] = Math.round(sumVal / Math.max(1e-6, sumW));
    }
  }

  return out;
}

/**
 * Sauvola local adaptive thresholding using integral images.
 * Produces binary output (0 = ink foreground, 255 = paper background)
 * resilient to shadows, uneven lighting, and faint strokes.
 */
export function sauvolaBinarize(
  gray: Uint8ClampedArray,
  w: number,
  h: number,
  windowRadius = 15,
  k = 0.2
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h);
  if (w === 0 || h === 0) return out;

  const integral = buildIntegralImage(gray, w, h);
  const sqIntegral = buildSquaredIntegralImage(gray, w, h);
  const R = 128;

  for (let y = 0; y < h; y++) {
    const y0 = y - windowRadius;
    const y1 = y + windowRadius;
    const rowOffset = y * w;
    for (let x = 0; x < w; x++) {
      const { sum, count } = boxSum(integral, w, h, x - windowRadius, y0, x + windowRadius, y1);
      const { sum: sqSum } = boxSum(sqIntegral, w, h, x - windowRadius, y0, x + windowRadius, y1);

      const mean = sum / count;
      const variance = Math.max(0, sqSum / count - mean * mean);
      const std = Math.sqrt(variance);
      const threshold = mean * (1 + k * (std / R - 1));

      out[rowOffset + x] = gray[rowOffset + x] < threshold ? 0 : 255;
    }
  }

  return out;
}

export const binarizeSauvola = sauvolaBinarize;
export const medianFilter = applyMedianFilter3x3;
export const gaussianBlur = applyGaussianBlur;

/**
 * Otsu's global adaptive thresholding maximizing inter-class variance between ink and background.
 * Fast and sharp for uniformly lit printed screenshots, PDFs, and flattened scans.
 */
export function binarizeOtsu(gray: Uint8ClampedArray, w: number, h: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h);
  if (w === 0 || h === 0) return out;

  const hist = new Int32Array(256);
  for (let i = 0; i < gray.length; i++) {
    hist[gray[i]]++;
  }

  const total = gray.length;
  let sumAll = 0;
  for (let t = 0; t < 256; t++) sumAll += t * hist[t];

  let sumB = 0;
  let wB = 0;
  let maxVar = -1;
  let threshold = 128;

  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;

    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sumAll - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > maxVar) {
      maxVar = between;
      threshold = t;
    }
  }

  for (let i = 0; i < gray.length; i++) {
    out[i] = gray[i] <= threshold ? 0 : 255;
  }
  return out;
}

/**
 * Binary morphological operations (0 = ink foreground, 255 = paper background):
 * - 'dilate': thickens/expands dark ink strokes by radius px
 * - 'erode': thins dark ink strokes & removes isolated dots
 * - 'close': dilate -> erode (bridges broken faded pen/pencil strokes)
 * - 'open': erode -> dilate (breaks thin noise bridges & bleeding characters)
 */
export function applyMorphologicalOp(
  binary: Uint8ClampedArray,
  w: number,
  h: number,
  op: 'dilate' | 'erode' | 'close' | 'open',
  radius = 1
): Uint8ClampedArray {
  const stepPass = (src: Uint8ClampedArray, mode: 'dilate' | 'erode'): Uint8ClampedArray => {
    const res = new Uint8ClampedArray(w * h);
    const fillVal = mode === 'dilate' ? 255 : 0;
    res.fill(fillVal);

    for (let y = 0; y < h; y++) {
      const y0 = Math.max(0, y - radius);
      const y1 = Math.min(h - 1, y + radius);
      const rowOff = y * w;
      for (let x = 0; x < w; x++) {
        const x0 = Math.max(0, x - radius);
        const x1 = Math.min(w - 1, x + radius);
        if (mode === 'dilate') {
          let hasInk = false;
          for (let ny = y0; ny <= y1 && !hasInk; ny++) {
            const nr = ny * w;
            for (let nx = x0; nx <= x1; nx++) {
              if (src[nr + nx] === 0) {
                hasInk = true;
                break;
              }
            }
          }
          res[rowOff + x] = hasInk ? 0 : 255;
        } else {
          let allInk = true;
          for (let ny = y0; ny <= y1 && allInk; ny++) {
            const nr = ny * w;
            for (let nx = x0; nx <= x1; nx++) {
              if (src[nr + nx] !== 0) {
                allInk = false;
                break;
              }
            }
          }
          res[rowOff + x] = allInk ? 0 : 255;
        }
      }
    }
    return res;
  };

  if (op === 'dilate' || op === 'erode') return stepPass(binary, op);
  if (op === 'close') return stepPass(stepPass(binary, 'dilate'), 'erode');
  return stepPass(stepPass(binary, 'erode'), 'dilate');
}

/**
 * Unified OCR preprocessing pipeline combining:
 * - Grayscale luminance conversion (RGBA/RGB/Gray)
 * - Dark-mode polarity inversion
 * - Optional perspective quad dewarping
 * - Local illumination flattening
 * - Noise reduction (3x3 Median + Gaussian / Bilateral smoothing)
 * - Adaptive Sauvola / Otsu binarization
 * - Optional deskewing & morphological cleanup
 */
export function preprocessForOcr(
  input: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
  options: PreprocessOptions = {}
): PreprocessResult {
  const appliedSteps: string[] = [];

  if (width <= 0 || height <= 0 || input.length === 0) {
    const empty = new Uint8ClampedArray(0);
    return {
      gray: empty,
      enhancedGray: empty,
      binary: empty,
      width,
      height,
      inverted: false,
      perspectiveWarped: false,
      skewAngle: 0,
      appliedSteps,
    };
  }

  let w = width;
  let h = height;
  let gray = toGrayscale(input, w, h, options.channels);
  appliedSteps.push('grayscale-luminance');

  let inverted = false;
  if (options.autoInvert !== false) {
    const invRes = autoInvertIfDarkMode(gray, w, h);
    gray = invRes.data;
    inverted = invRes.inverted;
    if (inverted) appliedSteps.push('color-inversion');
  }

  let perspectiveWarped = false;
  if (options.perspective) {
    const persp = correctDocumentPerspective(gray, w, h);
    if (persp.warped) {
      gray = persp.data;
      w = persp.width;
      h = persp.height;
      perspectiveWarped = true;
      appliedSteps.push('perspective-dewarp');
    }
  }

  if (options.flattenLighting !== false) {
    gray = flattenBackground(gray, w, h);
    appliedSteps.push('illumination-flatten');
  }

  const denoise = options.denoise ?? 'median+gaussian';
  const sigma = options.gaussianSigma ?? 0.85;

  if (denoise === 'auto') {
    const noiseLevel = estimateNoiseLevel(gray, w, h);
    gray = applyMedianFilter3x3(gray, w, h);
    appliedSteps.push('median-3x3');
    if (noiseLevel >= 4.0) {
      const autoSigma = options.gaussianSigma ?? 0.75;
      gray = applyGaussianBlur(gray, w, h, autoSigma, options.gaussianRadius);
      appliedSteps.push(`gaussian(sigma=${autoSigma})`);
    } else if (w >= 260 && h >= 180) {
      gray = applyBilateralSmoothing(gray, w, h, 1.8, 22);
      appliedSteps.push('bilateral-smooth');
    }
  } else {
    if (denoise === 'median' || denoise === 'median+gaussian' || denoise === 'median+bilateral') {
      gray = applyMedianFilter3x3(gray, w, h);
      appliedSteps.push('median-3x3');
    }
    if (denoise === 'gaussian' || denoise === 'median+gaussian') {
      gray = applyGaussianBlur(gray, w, h, sigma, options.gaussianRadius);
      appliedSteps.push(`gaussian(sigma=${sigma})`);
    }
    if (denoise === 'bilateral' || denoise === 'median+bilateral') {
      gray = applyBilateralSmoothing(gray, w, h, 1.8, 22);
      appliedSteps.push('bilateral-smooth');
    }
  }

  const winRadius =
    options.sauvolaWindowRadius ?? Math.max(12, Math.min(35, Math.round(Math.min(w, h) / 28)));
  const k = options.sauvolaK ?? (options.profile === 'printed' ? 0.18 : 0.2);
  const binarizeMode = options.binarize ?? 'sauvola';

  const computeBinary = (srcGray: Uint8ClampedArray): Uint8ClampedArray => {
    if (binarizeMode === 'otsu') {
      return binarizeOtsu(srcGray, w, h);
    }
    const sauvola = binarizeSauvola(srcGray, w, h, winRadius, k);
    if (binarizeMode === 'hybrid') {
      const otsu = binarizeOtsu(srcGray, w, h);
      const blended = new Uint8ClampedArray(w * h);
      for (let i = 0; i < blended.length; i++) {
        blended[i] = sauvola[i] === 0 || (otsu[i] === 0 && srcGray[i] < 135) ? 0 : 255;
      }
      return blended;
    }
    return sauvola;
  };

  let binary = computeBinary(gray);
  appliedSteps.push(
    binarizeMode === 'otsu'
      ? 'otsu-binarize'
      : binarizeMode === 'hybrid'
        ? 'sauvola-otsu-binarize'
        : 'sauvola-binarize'
  );

  const morph = options.morphology ?? 'none';
  if (morph === 'open' || morph === 'close') {
    binary = applyMorphologicalOp(binary, w, h, morph, 1);
    appliedSteps.push(`morph-${morph}`);
  }

  const enhancedGray =
    options.enhanceContrast !== false ? enhanceInkContrast(gray, w, h) : new Uint8ClampedArray(gray);
  if (options.enhanceContrast !== false) {
    appliedSteps.push('ink-contrast-stretch');
  }

  return {
    gray,
    enhancedGray,
    binary,
    width: w,
    height: h,
    inverted,
    perspectiveWarped,
    skewAngle: 0,
    appliedSteps,
  };
}

/**
 * Two-pass local multi-scale image enhancement with contour voting.
 * - Pass 1: Standard adaptive Sauvola thresholding capturing baseline layout and stroke bodies.
 * - Pass 2: High-contrast stroke sharpening with fine-structure recovery for thin or faded ink lines.
 * - Contour Voting: Performs consensus voting across both representations:
 *   - Retains ink pixels confirmed by both passes.
 *   - Retains fine stroke details with neighborhood contour support.
 *   - Suppresses isolated background salt-and-pepper noise and shadow gradients.
 */
export function twoPassContourVoting(
  gray: Uint8ClampedArray,
  w: number,
  h: number,
  options: { sauvolaWindowRadius?: number; sauvolaK?: number } = {}
): { binary: Uint8ClampedArray; enhancedGray: Uint8ClampedArray; inkRatio: number } {
  const winR = options.sauvolaWindowRadius ?? 16;
  const k1 = options.sauvolaK ?? 0.18;
  const pass1 = binarizeSauvola(gray, w, h, winR, k1);

  // Pass 2: High-pass Laplacian unsharp mask for recovering thin / faded pen strokes
  const sharpenedGray = new Uint8ClampedArray(w * h);
  sharpenedGray.set(gray);

  for (let y = 1; y < h - 1; y++) {
    const rowOff = y * w;
    for (let x = 1; x < w - 1; x++) {
      const c = gray[rowOff + x];
      const top = gray[rowOff - w + x];
      const bot = gray[rowOff + w + x];
      const left = gray[rowOff + x - 1];
      const right = gray[rowOff + x + 1];
      const laplacian = 5 * c - (top + bot + left + right);
      sharpenedGray[rowOff + x] = Math.max(0, Math.min(255, laplacian));
    }
  }

  const pass2 = binarizeSauvola(sharpenedGray, w, h, Math.max(8, Math.round(winR * 0.75)), 0.24);

  // Contour Consensus Voting
  const fused = new Uint8ClampedArray(w * h).fill(255);
  let totalInk = 0;

  for (let i = 0; i < w * h; i++) {
    const isInk1 = pass1[i] === 0;
    const isInk2 = pass2[i] === 0;

    if (isInk1 && isInk2) {
      fused[i] = 0;
      totalInk++;
    } else if (isInk1 || isInk2) {
      const x = i % w;
      const y = Math.floor(i / w);
      if (x > 0 && x < w - 1 && y > 0 && y < h - 1) {
        let neighborInkCount = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nIdx = (y + dy) * w + (x + dx);
            if (pass1[nIdx] === 0 || pass2[nIdx] === 0) {
              neighborInkCount++;
            }
          }
        }
        if (neighborInkCount >= 3) {
          fused[i] = 0;
          totalInk++;
        }
      }
    }
  }

  // Post-filter isolated speckles (noise pixels with zero adjacent ink neighbors)
  for (let y = 1; y < h - 1; y++) {
    const rowOff = y * w;
    for (let x = 1; x < w - 1; x++) {
      const idx = rowOff + x;
      if (fused[idx] === 0) {
        let neighborCount = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            if (fused[(y + dy) * w + (x + dx)] === 0) {
              neighborCount++;
            }
          }
        }
        if (neighborCount === 0) {
          fused[idx] = 255;
          totalInk--;
        }
      }
    }
  }

  const enhancedGray = enhanceInkContrast(gray, w, h);
  const inkRatio = totalInk / Math.max(1, w * h);

  return { binary: fused, enhancedGray, inkRatio };
}

/**
 * Convenience wrapper for browser Canvas `ImageData` (RGBA) objects.
 */
export function preprocessImageData(
  imageData: ImageDataLike,
  options: Omit<PreprocessOptions, 'channels'> = {}
): PreprocessResult {
  return preprocessForOcr(imageData.data, imageData.width, imageData.height, {
    ...options,
    channels: 4,
  });
}

/**
 * Retinex-based localized background illumination flattening and shadow removal.
 * Computes background estimation, normalizes lighting, and deepens text ink contrast.
 */
export function flattenDocumentLighting(
  rgba: Uint8ClampedArray,
  w: number,
  h: number
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h * 4);
  const gray = rgbaToGrayscale(rgba, w, h);
  const bg = flattenBackground(gray, w, h);
  const contrast = enhanceInkContrast(bg, w, h);

  for (let i = 0; i < w * h; i++) {
    const val = contrast[i];
    const off = i * 4;
    out[off] = val;
    out[off + 1] = val;
    out[off + 2] = val;
    out[off + 3] = 255;
  }
  return out;
}
