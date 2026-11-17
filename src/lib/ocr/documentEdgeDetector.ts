import {
  Point2D,
  QuadCorners,
  orderQuadCorners,
  detectDocumentQuad,
  warpPerspectiveRgba,
  estimateSkewAngle,
} from './geometry';
import { rgbaToGrayscale, binarizeSauvola } from './preprocess';

export interface NormalizedPoint {
  x: number; // 0..1
  y: number; // 0..1
}

export type NormalizedQuad = [
  NormalizedPoint,
  NormalizedPoint,
  NormalizedPoint,
  NormalizedPoint
];

export interface DocumentQuadDetectionResult {
  width: number;
  height: number;
  quad: QuadCorners;
  normalizedQuad: NormalizedQuad;
  autoDetected: boolean;
  mode: 'sheet' | 'text-block' | 'inset';
  confidence: number;
}

export interface ViewfinderAnalysis {
  isTextInFrame: boolean;
  textDensity: number; // 0..1
  framingQuality: 'optimal' | 'leveling-needed' | 'searching';
  estimatedTiltDeg: number;
  message: string;
}

/**
 * Detects text block bounding contours when the camera is zoomed in close (macro)
 * or when the paper sheet perimeter is cropped out of the viewfinder.
 */
export function detectTextBlockQuad(
  gray: Uint8ClampedArray,
  w: number,
  h: number
): QuadCorners | null {
  if (w < 80 || h < 60) return null;

  const binary = binarizeSauvola(gray, w, h, 14, 0.22);
  let minX = w;
  let maxX = 0;
  let minY = h;
  let maxY = 0;
  let inkCount = 0;

  // Scan with 3px step for performance
  for (let y = 8; y < h - 8; y += 3) {
    const rowOff = y * w;
    for (let x = 8; x < w - 8; x += 3) {
      if (binary[rowOff + x] === 0) {
        inkCount++;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  const spanW = maxX - minX;
  const spanH = maxY - minY;
  const totalSampled = ((w - 16) / 3) * ((h - 16) / 3);
  const inkRatio = inkCount / Math.max(1, totalSampled);

  // Require credible text ink density and non-trivial dimensions
  if (inkRatio < 0.015 || spanW < w * 0.2 || spanH < h * 0.12) {
    return null;
  }

  // Add 6% padding around the text block
  const padX = Math.round(w * 0.06);
  const padY = Math.round(h * 0.06);
  const x0 = Math.max(0, minX - padX);
  const y0 = Math.max(0, minY - padY);
  const x1 = Math.min(w - 1, maxX + padX);
  const y1 = Math.min(h - 1, maxY + padY);

  return [
    { x: x0, y: y0 },
    { x: x1, y: y0 },
    { x: x1, y: y1 },
    { x: x0, y: y1 },
  ];
}

/**
 * Real-time fast inspection of a downsampled viewfinder video frame.
 */
export function analyzeViewfinderFrame(
  canvas: HTMLCanvasElement
): ViewfinderAnalysis {
  const w = canvas.width;
  const h = canvas.height;
  if (w < 40 || h < 40) {
    return {
      isTextInFrame: false,
      textDensity: 0,
      framingQuality: 'searching',
      estimatedTiltDeg: 0,
      message: 'Position text within frame',
    };
  }

  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    return {
      isTextInFrame: false,
      textDensity: 0,
      framingQuality: 'searching',
      estimatedTiltDeg: 0,
      message: 'Position text within frame',
    };
  }

  const imgData = ctx.getImageData(0, 0, w, h);
  const gray = rgbaToGrayscale(imgData.data, w, h);
  const binary = binarizeSauvola(gray, w, h, 12, 0.2);

  let inkCount = 0;
  for (let i = 0; i < binary.length; i += 4) {
    if (binary[i] === 0) inkCount++;
  }
  const textDensity = inkCount / (binary.length / 4);
  const isTextInFrame = textDensity >= 0.02 && textDensity <= 0.45;

  let estimatedTiltDeg = 0;
  if (isTextInFrame) {
    estimatedTiltDeg = estimateSkewAngle(binary, w, h, { maxAngleDeg: 15, coarseStepDeg: 1 });
  }

  let framingQuality: 'optimal' | 'leveling-needed' | 'searching' = 'searching';
  let message = 'Align text or document inside frame';

  if (isTextInFrame) {
    if (Math.abs(estimatedTiltDeg) <= 3.5) {
      framingQuality = 'optimal';
      message = 'Text In Focus · Frame Aligned';
    } else {
      framingQuality = 'leveling-needed';
      message = `Tilt detected (${estimatedTiltDeg > 0 ? '+' : ''}${estimatedTiltDeg.toFixed(1)}°) · Hold Steady`;
    }
  }

  return {
    isTextInFrame,
    textDensity,
    framingQuality,
    estimatedTiltDeg,
    message,
  };
}

/**
 * Loads any image source string into an HTMLImageElement safely.
 */
function loadImage(source: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to load image for document edge detection.'));
    img.src = source;
  });
}

/**
 * Automatically detects the 4-corner document perimeter or close-up text block quadrilateral
 * on-device in the browser using adaptable boundary tracking.
 */
export async function detectDocumentPerimeter(
  source: string | HTMLCanvasElement
): Promise<DocumentQuadDetectionResult> {
  let origW = 0;
  let origH = 0;
  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D | null;

  if (typeof source === 'string') {
    const img = await loadImage(source);
    origW = img.naturalWidth || img.width;
    origH = img.naturalHeight || img.height;

    // Work on downsampled canvas for speed (< 30ms on mobile)
    const maxDim = 800;
    const scale = Math.min(1, maxDim / Math.max(origW, origH));
    const workW = Math.max(80, Math.round(origW * scale));
    const workH = Math.max(80, Math.round(origH * scale));

    canvas = document.createElement('canvas');
    canvas.width = workW;
    canvas.height = workH;
    ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('Canvas 2D context unavailable');
    ctx.drawImage(img, 0, 0, workW, workH);
  } else {
    origW = source.width;
    origH = source.height;
    canvas = source;
    ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('Canvas 2D context unavailable');
  }

  const workW = canvas.width;
  const workH = canvas.height;
  const imgData = ctx.getImageData(0, 0, workW, workH);
  const gray = rgbaToGrayscale(imgData.data, workW, workH);

  // 1. Try detecting full sheet perimeter against desk/background
  const detectedSheet = detectDocumentQuad(gray, workW, workH);

  let rawQuad: QuadCorners;
  let autoDetected = false;
  let mode: 'sheet' | 'text-block' | 'inset' = 'inset';
  let confidence = 0.5;

  if (detectedSheet) {
    const scaleX = origW / workW;
    const scaleY = origH / workH;
    rawQuad = [
      { x: Math.round(detectedSheet[0].x * scaleX), y: Math.round(detectedSheet[0].y * scaleY) },
      { x: Math.round(detectedSheet[1].x * scaleX), y: Math.round(detectedSheet[1].y * scaleY) },
      { x: Math.round(detectedSheet[2].x * scaleX), y: Math.round(detectedSheet[2].y * scaleY) },
      { x: Math.round(detectedSheet[3].x * scaleX), y: Math.round(detectedSheet[3].y * scaleY) },
    ];
    autoDetected = true;
    mode = 'sheet';
    confidence = 0.92;
  } else {
    // 2. Adaptable fallback for close-up / angled shots: detect tight text-block hull
    const detectedTextBlock = detectTextBlockQuad(gray, workW, workH);
    if (detectedTextBlock) {
      const scaleX = origW / workW;
      const scaleY = origH / workH;
      rawQuad = [
        { x: Math.round(detectedTextBlock[0].x * scaleX), y: Math.round(detectedTextBlock[0].y * scaleY) },
        { x: Math.round(detectedTextBlock[1].x * scaleX), y: Math.round(detectedTextBlock[1].y * scaleY) },
        { x: Math.round(detectedTextBlock[2].x * scaleX), y: Math.round(detectedTextBlock[2].y * scaleY) },
        { x: Math.round(detectedTextBlock[3].x * scaleX), y: Math.round(detectedTextBlock[3].y * scaleY) },
      ];
      autoDetected = true;
      mode = 'text-block';
      confidence = 0.86;
    } else {
      // 3. Fallback perimeter inset slightly from camera frame
      const insetX = Math.round(origW * 0.04);
      const insetY = Math.round(origH * 0.04);
      rawQuad = [
        { x: insetX, y: insetY },
        { x: origW - insetX, y: insetY },
        { x: origW - insetX, y: origH - insetY },
        { x: insetX, y: origH - insetY },
      ];
      mode = 'inset';
    }
  }

  const ordered = orderQuadCorners(rawQuad);
  const normalizedQuad: NormalizedQuad = [
    { x: Math.max(0, Math.min(1, ordered[0].x / origW)), y: Math.max(0, Math.min(1, ordered[0].y / origH)) },
    { x: Math.max(0, Math.min(1, ordered[1].x / origW)), y: Math.max(0, Math.min(1, ordered[1].y / origH)) },
    { x: Math.max(0, Math.min(1, ordered[2].x / origW)), y: Math.max(0, Math.min(1, ordered[2].y / origH)) },
    { x: Math.max(0, Math.min(1, ordered[3].x / origW)), y: Math.max(0, Math.min(1, ordered[3].y / origH)) },
  ];

  return {
    width: origW,
    height: origH,
    quad: ordered,
    normalizedQuad,
    autoDetected,
    mode,
    confidence,
  };
}

/**
 * Dewarps a quadrilateral region from an image into an upright, flat rectangular image
 * using 8-DOF planar projective homography and bilinear texture remapping.
 */
export async function dewarpImageWithQuad(
  imageSource: string,
  quad: QuadCorners,
  options: { maxWidth?: number; maxHeight?: number } = {}
): Promise<{ dataUrl: string; width: number; height: number }> {
  const img = await loadImage(imageSource);
  const origW = img.naturalWidth || img.width;
  const origH = img.naturalHeight || img.height;

  // Render original full image onto offscreen canvas to obtain raw RGBA
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = origW;
  srcCanvas.height = origH;
  const srcCtx = srcCanvas.getContext('2d', { willReadFrequently: true });
  if (!srcCtx) throw new Error('Canvas 2D context unavailable');
  srcCtx.drawImage(img, 0, 0, origW, origH);
  const srcData = srcCtx.getImageData(0, 0, origW, origH);

  // Compute desired destination dimensions from quad edges
  const ordered = orderQuadCorners(quad);
  const topW = Math.hypot(ordered[1].x - ordered[0].x, ordered[1].y - ordered[0].y);
  const botW = Math.hypot(ordered[2].x - ordered[3].x, ordered[2].y - ordered[3].y);
  const leftH = Math.hypot(ordered[3].x - ordered[0].x, ordered[3].y - ordered[0].y);
  const rightH = Math.hypot(ordered[2].x - ordered[1].x, ordered[2].y - ordered[1].y);

  let targetW = Math.round(Math.max(topW, botW));
  let targetH = Math.round(Math.max(leftH, rightH));

  const maxW = options.maxWidth ?? 1600;
  const maxH = options.maxHeight ?? 1600;
  if (targetW > maxW || targetH > maxH) {
    const scale = Math.min(maxW / targetW, maxH / targetH);
    targetW = Math.round(targetW * scale);
    targetH = Math.round(targetH * scale);
  }

  const warped = warpPerspectiveRgba(srcData.data, origW, origH, ordered, {
    targetWidth: targetW,
    targetHeight: targetH,
  });

  const dstCanvas = document.createElement('canvas');
  dstCanvas.width = warped.width;
  dstCanvas.height = warped.height;
  const dstCtx = dstCanvas.getContext('2d');
  if (!dstCtx) throw new Error('Canvas 2D context unavailable');

  const dstImgData = dstCtx.createImageData(warped.width, warped.height);
  dstImgData.data.set(warped.data);
  dstCtx.putImageData(dstImgData, 0, 0);

  const dataUrl = dstCanvas.toDataURL('image/jpeg', 0.93);
  return { dataUrl, width: warped.width, height: warped.height };
}
