import {
  Point2D,
  QuadCorners,
  orderQuadCorners,
  detectDocumentQuad,
  warpPerspectiveRgba,
} from './geometry';
import { rgbaToGrayscale } from './preprocess';

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
  confidence: number;
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
 * Automatically detects the 4-corner document perimeter quadrilateral from an image
 * on-device in the browser using edge-luminance boundary tracking.
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

  const detected = detectDocumentQuad(gray, workW, workH);

  let rawQuad: QuadCorners;
  let autoDetected = false;
  let confidence = 0.5;

  if (detected) {
    const scaleX = origW / workW;
    const scaleY = origH / workH;
    rawQuad = [
      { x: Math.round(detected[0].x * scaleX), y: Math.round(detected[0].y * scaleY) },
      { x: Math.round(detected[1].x * scaleX), y: Math.round(detected[1].y * scaleY) },
      { x: Math.round(detected[2].x * scaleX), y: Math.round(detected[2].y * scaleY) },
      { x: Math.round(detected[3].x * scaleX), y: Math.round(detected[3].y * scaleY) },
    ];
    autoDetected = true;
    confidence = 0.88;
  } else {
    // Smart default perimeter inset slightly from camera frame
    const insetX = Math.round(origW * 0.04);
    const insetY = Math.round(origH * 0.04);
    rawQuad = [
      { x: insetX, y: insetY },
      { x: origW - insetX, y: insetY },
      { x: origW - insetX, y: origH - insetY },
      { x: insetX, y: origH - insetY },
    ];
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
