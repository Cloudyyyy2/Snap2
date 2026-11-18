import {
  Point2D,
  QuadCorners,
  orderQuadCorners,
  detectDocumentQuad,
  warpPerspectiveRgba,
} from './geometry';
import { rgbaToGrayscale, binarizeSauvola, flattenDocumentLighting } from './preprocess';
import { segmentLines, LineBox } from './handwriting/lines';
import { removeRuledLines } from './handwriting/cleanup';
import { preprocessForOCR } from '../../services/imageProcessor';

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

export interface DetectedTextLine {
  id: string;
  lineIndex: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  normalized: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

export interface InCropperTextDetection {
  width: number;
  height: number;
  lines: DetectedTextLine[];
  textHullQuad: QuadCorners | null;
  normalizedTextHull: NormalizedQuad | null;
  totalInkPixels: number;
}

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
  framingQuality: 'optimal' | 'searching';
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
      message: 'Position text within frame',
    };
  }

  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    return {
      isTextInFrame: false,
      textDensity: 0,
      framingQuality: 'searching',
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

  const framingQuality: 'optimal' | 'searching' = isTextInFrame ? 'optimal' : 'searching';
  const message = isTextInFrame ? 'Text In Focus · Frame Aligned' : 'Align text or document inside frame';

  return {
    isTextInFrame,
    textDensity,
    framingQuality,
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

  // Apply high-performance adaptive thresholding & shadow removal for robust cropping
  const enhancedImageData = preprocessForOCR(canvas);
  const gray = rgbaToGrayscale(enhancedImageData.data, workW, workH);

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
  options: {
    maxWidth?: number;
    maxHeight?: number;
    enhanceDocLighting?: boolean;
  } = {}
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

  // Modern OCR 300 DPI standard target (default up to 2200px)
  const maxW = options.maxWidth ?? 2200;
  const maxH = options.maxHeight ?? 2200;
  if (targetW > maxW || targetH > maxH) {
    const scale = Math.min(maxW / targetW, maxH / targetH);
    targetW = Math.round(targetW * scale);
    targetH = Math.round(targetH * scale);
  }

  const warped = warpPerspectiveRgba(srcData.data, origW, origH, ordered, {
    targetWidth: targetW,
    targetHeight: targetH,
  });

  const outputData = options.enhanceDocLighting
    ? flattenDocumentLighting(warped.data, warped.width, warped.height)
    : warped.data;

  const dstCanvas = document.createElement('canvas');
  dstCanvas.width = warped.width;
  dstCanvas.height = warped.height;
  const dstCtx = dstCanvas.getContext('2d');
  if (!dstCtx) throw new Error('Canvas 2D context unavailable');

  const dstImgData = dstCtx.createImageData(warped.width, warped.height);
  dstImgData.data.set(outputData);
  dstCtx.putImageData(dstImgData, 0, 0);

  const dataUrl = dstCanvas.toDataURL('image/jpeg', 0.94);
  return { dataUrl, width: warped.width, height: warped.height };
}

/**
 * Standard ray-casting point-in-polygon containment algorithm.
 */
export function isPointInPolygon(pt: Point2D, poly: Point2D[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x;
    const yi = poly[i].y;
    const xj = poly[j].x;
    const yj = poly[j].y;
    const intersect =
      yi > pt.y !== yj > pt.y &&
      pt.x < ((xj - xi) * (pt.y - yi)) / (yj - yi + 1e-9) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

/**
 * Checks whether a normalized rectangular text bounding box is safely enclosed
 * within a 4-point polygon (e.g. crop quad).
 */
export function isBoxInPolygon(
  box: { x: number; y: number; width: number; height: number },
  poly: Point2D[]
): boolean {
  const corners: Point2D[] = [
    { x: box.x, y: box.y },
    { x: box.x + box.width, y: box.y },
    { x: box.x + box.width, y: box.y + box.height },
    { x: box.x, y: box.y + box.height },
  ];
  for (const c of corners) {
    if (!isPointInPolygon(c, poly)) return false;
  }
  return true;
}

/**
 * Detects all visible text lines on the raw un-cropped image inside the cropper (< 25ms),
 * returning glowing boundary ribbons for every word/line and the tight text hull quad.
 */
export async function detectTextLinesAndHull(
  source: string | HTMLCanvasElement
): Promise<InCropperTextDetection> {
  let origW = 0;
  let origH = 0;
  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D | null;

  if (typeof source === 'string') {
    const img = await loadImage(source);
    origW = img.naturalWidth || img.width;
    origH = img.naturalHeight || img.height;

    const maxDim = 900;
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

  // Erase notebook ruling lines so empty lines aren't treated as text
  const noRules = removeRuledLines(gray, workW, workH);
  const binary = binarizeSauvola(noRules, workW, workH, 14, 0.22);

  // Extract text line boxes
  const rawLines: LineBox[] = segmentLines(binary, workW, workH);
  const scaleX = origW / workW;
  const scaleY = origH / workH;

  let totalInk = 0;
  for (let i = 0; i < binary.length; i++) {
    if (binary[i] === 0) totalInk++;
  }

  const detectedLines: DetectedTextLine[] = rawLines.map((l, idx) => {
    const origX0 = Math.round(l.x0 * scaleX);
    const origY0 = Math.round(l.y0 * scaleY);
    const origX1 = Math.round(l.x1 * scaleX);
    const origY1 = Math.round(l.y1 * scaleY);

    return {
      id: `line-${idx}-${Date.now()}`,
      lineIndex: idx,
      x0: origX0,
      y0: origY0,
      x1: origX1,
      y1: origY1,
      normalized: {
        x: Math.max(0, Math.min(1, origX0 / origW)),
        y: Math.max(0, Math.min(1, origY0 / origH)),
        width: Math.max(0.01, Math.min(1, (origX1 - origX0) / origW)),
        height: Math.max(0.01, Math.min(1, (origY1 - origY0) / origH)),
      },
    };
  });

  let textHullQuad: QuadCorners | null = null;
  let normalizedTextHull: NormalizedQuad | null = null;

  if (detectedLines.length > 0) {
    let minX = origW;
    let minY = origH;
    let maxX = 0;
    let maxY = 0;

    for (const l of detectedLines) {
      if (l.x0 < minX) minX = l.x0;
      if (l.y0 < minY) minY = l.y0;
      if (l.x1 > maxX) maxX = l.x1;
      if (l.y1 > maxY) maxY = l.y1;
    }

    // Add 12% comfortable breathing margin around the text cluster (prevents cutting off the first letter)
    const padX = Math.round(origW * 0.12);
    const padY = Math.round(origH * 0.10);

    const x0 = Math.max(0, minX - padX);
    const y0 = Math.max(0, minY - padY);
    const x1 = Math.min(origW - 1, maxX + padX);
    const y1 = Math.min(origH - 1, maxY + padY);

    textHullQuad = [
      { x: x0, y: y0 },
      { x: x1, y: y0 },
      { x: x1, y: y1 },
      { x: x0, y: y1 },
    ];

    normalizedTextHull = [
      { x: x0 / origW, y: y0 / origH },
      { x: x1 / origW, y: y0 / origH },
      { x: x1 / origW, y: y1 / origH },
      { x: x0 / origW, y: y1 / origH },
    ];
  }

  return {
    width: origW,
    height: origH,
    lines: detectedLines,
    textHullQuad,
    normalizedTextHull,
    totalInkPixels: totalInk,
  };
}

/**
 * Rotates an image Data URL by 90°, 180°, or 270° clockwise with pristine canvas rendering.
 */
export async function rotateImageBase64(
  imageSource: string,
  angleDeg: number
): Promise<string> {
  const normAngle = ((angleDeg % 360) + 360) % 360;
  if (normAngle === 0) return imageSource;

  const img = await loadImage(imageSource);
  const origW = img.naturalWidth || img.width;
  const origH = img.naturalHeight || img.height;

  const isPerpendicular = normAngle === 90 || normAngle === 270;
  const canvasW = isPerpendicular ? origH : origW;
  const canvasH = isPerpendicular ? origW : origH;

  const canvas = document.createElement('canvas');
  canvas.width = canvasW;
  canvas.height = canvasH;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D context unavailable');

  ctx.translate(canvasW / 2, canvasH / 2);
  ctx.rotate((normAngle * Math.PI) / 180);
  ctx.drawImage(img, -origW / 2, -origH / 2);

  return canvas.toDataURL('image/jpeg', 0.94);
}

export interface LiveTrackedLine {
  id: string;
  x: number;      // normalized 0..1
  y: number;      // normalized 0..1
  width: number;  // normalized 0..1
  height: number; // normalized 0..1
}

export interface LiveTrackedTextResult {
  lines: LiveTrackedLine[];
  lineCount: number;
  textHullBox?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  isTextInFrame: boolean;
  textDensity: number;
}

/**
 * Ultra-fast (< 10ms) live text line and word tracker running continuously over downsampled camera frames.
 * Powers real-time glowing bounding ribbons directly in the active viewfinder.
 */
export function trackLiveTextLines(
  canvas: HTMLCanvasElement,
  prevLines: LiveTrackedLine[] = []
): LiveTrackedTextResult {
  const w = canvas.width;
  const h = canvas.height;
  if (w < 40 || h < 40) {
    return {
      lines: [],
      lineCount: 0,
      isTextInFrame: false,
      textDensity: 0,
    };
  }

  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    return {
      lines: [],
      lineCount: 0,
      isTextInFrame: false,
      textDensity: 0,
    };
  }

  const imgData = ctx.getImageData(0, 0, w, h);
  const gray = rgbaToGrayscale(imgData.data, w, h);
  const binary = binarizeSauvola(gray, w, h, 12, 0.22);

  let inkCount = 0;
  for (let i = 0; i < binary.length; i += 4) {
    if (binary[i] === 0) inkCount++;
  }
  const textDensity = inkCount / (binary.length / 4);

  // Require credible ink presence (ignore solid white or solid black frames)
  if (textDensity < 0.015 || textDensity > 0.5) {
    return {
      lines: [],
      lineCount: 0,
      isTextInFrame: false,
      textDensity,
    };
  }

  // Segment candidate text lines
  const rawLines = segmentLines(binary, w, h);
  if (rawLines.length === 0) {
    return {
      lines: [],
      lineCount: 0,
      isTextInFrame: false,
      textDensity,
    };
  }

  const rawTracked: LiveTrackedLine[] = rawLines
    .filter((l) => {
      const bw = l.x1 - l.x0;
      const bh = l.y1 - l.y0;
      // Filter out tiny 2px noise specks or page-border full screen bars
      return bw >= w * 0.06 && bh >= 6 && bh <= h * 0.45;
    })
    .map((l, idx) => ({
      id: `live-line-${idx}`,
      x: Math.max(0, Math.min(1, l.x0 / w)),
      y: Math.max(0, Math.min(1, l.y0 / h)),
      width: Math.max(0.02, Math.min(1, (l.x1 - l.x0) / w)),
      height: Math.max(0.02, Math.min(1, (l.y1 - l.y0) / h)),
    }));

  // Exponential moving average (EMA) smoothing against prevLines to eliminate frame-to-frame jitter
  const smoothedLines = rawTracked.map((curr) => {
    const match = prevLines.find((prev) => {
      const vOverlap = Math.max(
        0,
        Math.min(prev.y + prev.height, curr.y + curr.height) - Math.max(prev.y, curr.y)
      );
      return vOverlap > curr.height * 0.35;
    });

    if (match) {
      return {
        id: match.id,
        x: Number((match.x * 0.35 + curr.x * 0.65).toFixed(4)),
        y: Number((match.y * 0.35 + curr.y * 0.65).toFixed(4)),
        width: Number((match.width * 0.35 + curr.width * 0.65).toFixed(4)),
        height: Number((match.height * 0.35 + curr.height * 0.65).toFixed(4)),
      };
    }
    return curr;
  });

  let textHullBox: { x: number; y: number; width: number; height: number } | undefined;
  if (smoothedLines.length > 0) {
    let minX = 1;
    let minY = 1;
    let maxX = 0;
    let maxY = 0;

    for (const l of smoothedLines) {
      if (l.x < minX) minX = l.x;
      if (l.y < minY) minY = l.y;
      if (l.x + l.width > maxX) maxX = l.x + l.width;
      if (l.y + l.height > maxY) maxY = l.y + l.height;
    }

    const padX = 0.06;
    const padY = 0.06;
    const hx = Math.max(0, minX - padX);
    const hy = Math.max(0, minY - padY);
    const hw = Math.min(1 - hx, maxX - minX + padX * 2);
    const hh = Math.min(1 - hy, maxY - minY + padY * 2);

    textHullBox = { x: hx, y: hy, width: hw, height: hh };
  }

  return {
    lines: smoothedLines,
    lineCount: smoothedLines.length,
    textHullBox,
    isTextInFrame: smoothedLines.length > 0,
    textDensity,
  };
}



