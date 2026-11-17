/**
 * Pure TypeScript grayscale image cleanup for everyday handwriting on paper.
 * Operates on Uint8ClampedArray (1 byte per pixel, 0 = black ink, 255 = white paper),
 * usable identically in both Node.js and browser environments.
 */

export {
  type Point2D,
  type QuadCorners,
  type DeskewOptions,
  type DeskewResult,
  type PerspectiveWarpOptions,
  type PerspectiveResult,
  orderQuadCorners,
  solveHomography,
  applyHomographyPoint,
  warpPerspectiveQuad,
  detectDocumentQuad,
  estimateSkewAngle,
  rotateGrayscale,
  deskewGrayscale,
  correctDocumentPerspective,
} from '../geometry';

export {
  type ChannelCount,
  type DenoiseMethod,
  type BinarizeMethod,
  type PreprocessOptions,
  type PreprocessResult,
  type ImageDataLike,
  rgbaToGrayscale,
  rgbToGrayscale,
  toGrayscale,
  grayscaleToRgba,
  autoInvertIfDarkMode,
  flattenBackground,
  enhanceInkContrast,
  applyMedianFilter3x3,
  medianFilter,
  applyGaussianBlur,
  gaussianBlur,
  applyBilateralSmoothing,
  estimateNoiseLevel,
  binarizeSauvola,
  sauvolaBinarize,
  binarizeOtsu,
  applyMorphologicalOp,
  preprocessForOcr,
  preprocessImageData,
} from '../preprocess';

/**
 * Detects thin horizontal notebook ruled lines (long horizontal runs > 60% width,
 * thickness <= 3 px) and paints them with the local paper brightness while keeping
 * darker pen strokes that cross the ruled lines intact.
 */
export function removeRuledLines(
  gray: Uint8ClampedArray,
  w: number,
  h: number
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(gray);
  if (w < 20 || h < 10) return out;

  // Estimate paper background per row using the 85th percentile sample across width,
  // then take the vertical local maximum across +/- 5 rows so full-width ruled lines don't mask themselves.
  const rawRowPaper = new Uint8Array(h);
  const sampleStep = Math.max(1, Math.floor(w / 64));
  const samples: number[] = [];

  for (let y = 0; y < h; y++) {
    samples.length = 0;
    const rowOffset = y * w;
    for (let x = 0; x < w; x += sampleStep) {
      samples.push(gray[rowOffset + x]);
    }
    samples.sort((a, b) => a - b);
    rawRowPaper[y] = samples[Math.floor(samples.length * 0.85)] ?? 245;
  }

  const rowPaper = new Uint8Array(h);
  for (let y = 0; y < h; y++) {
    let maxBg = rawRowPaper[y];
    for (let dy = -5; dy <= 5; dy++) {
      const ny = y + dy;
      if (ny >= 0 && ny < h && rawRowPaper[ny] > maxBg) {
        maxBg = rawRowPaper[ny];
      }
    }
    rowPaper[y] = maxBg;
  }

  // Flag candidate rows that have a consistent horizontal gray band (> 55% of pixels darker than paper - 10,
  // yet not heavy dark handwriting ink across the whole width)
  const candidateRow = new Uint8Array(h);
  const rowMedian = new Uint8Array(h);

  for (let y = 0; y < h; y++) {
    const paper = rowPaper[y];
    const rowOffset = y * w;
    let covered = 0;
    let maxRun = 0;
    let curRun = 0;
    const lineSamples: number[] = [];

    for (let x = 0; x < w; x++) {
      const v = gray[rowOffset + x];
      if (v <= paper - 10) {
        covered++;
        curRun++;
        if (curRun > maxRun) maxRun = curRun;
        if (x % sampleStep === 0) lineSamples.push(v);
      } else {
        curRun = 0;
      }
    }

    if (covered / w >= 0.55 && maxRun >= w * 0.3 && lineSamples.length > 0) {
      lineSamples.sort((a, b) => a - b);
      const med = lineSamples[Math.floor(lineSamples.length * 0.5)];
      // Ruled lines are typically mid/light gray/blue (not solid black ink block)
      if (med >= 55 && med <= paper - 10) {
        candidateRow[y] = 1;
        rowMedian[y] = med;
      }
    }
  }

  // Group contiguous candidate rows; ruled lines have thickness <= 4 px
  let y = 0;
  while (y < h) {
    if (!candidateRow[y]) {
      y++;
      continue;
    }

    const startY = y;
    while (y < h && candidateRow[y]) {
      y++;
    }
    const thickness = y - startY;

    if (thickness <= 4) {
      // Estimate surrounding non-line paper brightness above/below the band
      const aboveY = Math.max(0, startY - 2);
      const belowY = Math.min(h - 1, y + 1);

      for (let ry = startY; ry < y; ry++) {
        const lineTone = rowMedian[ry];
        const rowOffset = ry * w;
        for (let x = 0; x < w; x++) {
          const cur = out[rowOffset + x];
          const surround = Math.round((gray[aboveY * w + x] + gray[belowY * w + x]) / 2);
          // Keep crossing ink strokes that are noticeably darker than the ruled line
          const isCrossingInk = cur < lineTone - 24;
          if (!isCrossingInk && cur <= surround - 6) {
            out[rowOffset + x] = Math.max(cur, surround, rowPaper[ry]);
          }
        }
      }
    }
  }

  return out;
}



/**
 * Filters out non-text background clutter (desks, room shadows, fingers, spiral bindings,
 * long dividers, and 1-2px speckles) from a Sauvola binary buffer (0 = ink, 255 = paper).
 */
export function filterTextComponents(
  binary: Uint8ClampedArray,
  w: number,
  h: number
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h);
  out.fill(255);
  if (w === 0 || h === 0) return out;

  const totalPixels = w * h;
  const visited = new Uint8Array(totalPixels);
  const queue: number[] = [];
  const compPixels: number[] = [];
  const edgeMarginX = Math.max(4, Math.round(w * 0.02));
  const edgeMarginY = Math.max(4, Math.round(h * 0.02));

  for (let startIdx = 0; startIdx < totalPixels; startIdx++) {
    if (visited[startIdx] || binary[startIdx] !== 0) continue;

    visited[startIdx] = 1;
    queue.length = 0;
    compPixels.length = 0;
    queue.push(startIdx);

    let head = 0;
    let minX = w;
    let maxX = -1;
    let minY = h;
    let maxY = -1;

    while (head < queue.length) {
      const idx = queue[head++];
      compPixels.push(idx);

      const y = Math.floor(idx / w);
      const x = idx - y * w;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;

      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= h) continue;
        const rowOff = ny * w;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= w) continue;
          const nIdx = rowOff + nx;
          if (!visited[nIdx] && binary[nIdx] === 0) {
            visited[nIdx] = 1;
            queue.push(nIdx);
          }
        }
      }
    }

    const cw = maxX - minX + 1;
    const ch = maxY - minY + 1;
    const area = compPixels.length;
    const touchesEdge =
      minX <= edgeMarginX ||
      maxX >= w - 1 - edgeMarginX ||
      minY <= edgeMarginY ||
      maxY >= h - 1 - edgeMarginY;

    // Reject tiny noise specks
    if (area < 5 || (cw <= 2 && ch <= 2)) continue;

    // Reject massive solid blobs / desks / hands / shadows
    if (area > totalPixels * 0.02) continue;
    if (cw > w * 0.55 && ch > h * 0.18) continue;
    if (ch > h * 0.38) continue;

    // Reject large edge-touching clutter (hands holding paper, desk edges, spiral coils)
    if (touchesEdge && (ch > h * 0.14 || cw > w * 0.28 || area > totalPixels * 0.006)) {
      continue;
    }

    // Reject long thin horizontal/vertical border rules or spiral spines
    if (cw > w * 0.65 && ch <= 5) continue;
    if (ch > h * 0.45 && cw <= 6) continue;

    for (let i = 0; i < compPixels.length; i++) {
      out[compPixels[i]] = 0;
    }
  }

  return out;
}



export interface DocumentCropResult {
  data: Uint8ClampedArray;
  width: number;
  height: number;
  cropped: boolean;
  bounds: { x0: number; y0: number; x1: number; y1: number };
}

/**
 * Detects dark outer background clutter, scanner frames, hands, or spiral edges surrounding
 * a bright paper/page sheet and crops cleanly to the paper interior before line segmentation.
 */
export function cropDarkBordersAndMargins(
  gray: Uint8ClampedArray,
  w: number,
  h: number
): DocumentCropResult {
  if (w < 120 || h < 80) {
    return {
      data: new Uint8ClampedArray(gray),
      width: w,
      height: h,
      cropped: false,
      bounds: { x0: 0, y0: 0, x1: w - 1, y1: h - 1 },
    };
  }

  // Measure per-column and per-row bright-paper coverage (>= 175)
  const colBrightRatio = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    let bright = 0;
    for (let y = 0; y < h; y += 2) {
      if (gray[y * w + x] >= 175) bright++;
    }
    colBrightRatio[x] = bright / Math.ceil(h / 2);
  }

  const rowBrightRatio = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    let bright = 0;
    const rowOff = y * w;
    for (let x = 0; x < w; x += 2) {
      if (gray[rowOff + x] >= 175) bright++;
    }
    rowBrightRatio[y] = bright / Math.ceil(w / 2);
  }

  // Find longest contiguous horizontal paper span where colBrightRatio >= 0.58
  let bestX0 = 0;
  let bestX1 = w - 1;
  let curStart = -1;
  let bestLen = 0;

  for (let x = 0; x < w; x++) {
    if (colBrightRatio[x] >= 0.68) {
      if (curStart === -1) curStart = x;
      const len = x - curStart + 1;
      if (len > bestLen) {
        bestLen = len;
        bestX0 = curStart;
        bestX1 = x;
      }
    } else {
      curStart = -1;
    }
  }

  // Only crop horizontally if a clear page sheet exists (>= 30% width) and dark borders were present
  let x0 = 0;
  let x1 = w - 1;
  if (bestLen >= w * 0.3 && bestLen <= w * 0.94) {
    const padX = Math.max(4, Math.round(w * 0.01));
    x0 = Math.min(w - 1, bestX0 + padX);
    x1 = Math.max(x0 + 1, bestX1 - padX);
  }

  // Trim dark outer top/bottom scanner frames inside [x0..x1]
  let y0 = 0;
  let y1 = h - 1;
  while (y0 < h * 0.25 && rowBrightRatio[y0] < 0.4) y0++;
  while (y1 > h * 0.75 && rowBrightRatio[y1] < 0.4) y1--;

  const cropW = x1 - x0 + 1;
  const cropH = y1 - y0 + 1;
  const changed = cropW < w * 0.96 || cropH < h * 0.96;

  if (!changed || cropW < 60 || cropH < 40) {
    return {
      data: new Uint8ClampedArray(gray),
      width: w,
      height: h,
      cropped: false,
      bounds: { x0: 0, y0: 0, x1: w - 1, y1: h - 1 },
    };
  }

  const out = new Uint8ClampedArray(cropW * cropH);
  for (let y = 0; y < cropH; y++) {
    const srcOff = (y0 + y) * w + x0;
    const dstOff = y * cropW;
    for (let x = 0; x < cropW; x++) {
      out[dstOff + x] = gray[srcOff + x];
    }
  }

  return {
    data: out,
    width: cropW,
    height: cropH,
    cropped: true,
    bounds: { x0, y0, x1, y1 },
  };
}

/**
 * Computes recommended resolution upsampling factor (1.0 to 2.5) to bring small UI fonts
 * or compressed document text toward an effective 300-450 DPI recognition scale.
 */
export function computeRecommendedDpiScale(
  binary: Uint8ClampedArray,
  w: number,
  h: number
): number {
  if (w === 0 || h === 0) return 1;

  // Measure horizontal text-line band heights via row projection so thin strokes on large
  // handwriting (e.g. 60px letters with 5px pen width) are not mistaken for tiny 5px fonts.
  const bandHeights: number[] = [];
  let inBand = false;
  let bandStart = 0;
  const minRowInk = Math.max(3, Math.round(w * 0.005));

  for (let y = 0; y < h; y++) {
    let rowInk = 0;
    const off = y * w;
    for (let x = 0; x < w; x++) {
      if (binary[off + x] === 0) rowInk++;
    }
    if (rowInk >= minRowInk) {
      if (!inBand) {
        inBand = true;
        bandStart = y;
      }
    } else if (inBand) {
      const bh = y - bandStart;
      if (bh >= 5 && bh <= h * 0.45) bandHeights.push(bh);
      inBand = false;
    }
  }
  if (inBand) {
    const bh = h - bandStart;
    if (bh >= 5 && bh <= h * 0.45) bandHeights.push(bh);
  }

  if (bandHeights.length === 0) return w < 640 ? 1.5 : 1.0;
  bandHeights.sort((a, b) => a - b);
  const p75BandH = bandHeights[Math.floor(bandHeights.length * 0.75)];

  // Target >= 26px text band height (~300 DPI equivalent)
  if (p75BandH < 12) return 2.5;
  if (p75BandH < 18) return 2.0;
  if (p75BandH < 26) return 1.5;
  return 1.0;
}
