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
  type DeslantOptions,
  type DeslantResult,
  orderQuadCorners,
  solveHomography,
  applyHomographyPoint,
  warpPerspectiveQuad,
  detectDocumentQuad,
  estimateSkewAngle,
  rotateGrayscale,
  deskewGrayscale,
  deslantGrayscale,
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
  twoPassContourVoting,
} from '../preprocess';

export {
  type LineBox,
  type CroppedLine,
  type WordBox,
  type CroppedSquare,
  segmentLines,
  segmentWords,
  splitWideLineBox,
  cropLine,
  cropToSquare,
} from './lines';

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

    if (covered / w >= 0.35 && maxRun >= w * 0.20 && lineSamples.length > 0) {
      lineSamples.sort((a, b) => a - b);
      const med = lineSamples[Math.floor(lineSamples.length * 0.5)];
      // Ruled lines are typically mid/light gray/blue (not solid black ink block)
      if (med >= 45 && med <= paper - 8) {
        candidateRow[y] = 1;
        rowMedian[y] = med;
      }
    }
  }

  // Group contiguous candidate rows; ruled lines have thickness <= 6 px (scaled to width)
  const maxAllowedThickness = Math.max(5, Math.round(w / 280));
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

    if (thickness <= maxAllowedThickness) {
      // Estimate surrounding non-line paper brightness above/below the band
      const aboveY = Math.max(0, startY - 2);
      const belowY = Math.min(h - 1, y + 1);

      for (let ry = startY; ry < y; ry++) {
        const lineTone = rowMedian[ry];
        const paper = rowPaper[ry];
        const rowOffset = ry * w;
        for (let x = 0; x < w; x++) {
          const cur = out[rowOffset + x];
          const surround = Math.round((gray[aboveY * w + x] + gray[belowY * w + x]) / 2);

          // Check if pixel belongs to a vertical handwriting stroke (graphite pencil or pen crossing the ruling)
          const hasVerticalAbove = ry >= 3 && gray[(ry - 2) * w + x] < paper - 14;
          const hasVerticalBelow = ry < h - 3 && gray[(ry + 2) * w + x] < paper - 14;
          const isNoticeablyDarker = cur < lineTone - 14;
          const isCrossingInk = isNoticeablyDarker || hasVerticalAbove || hasVerticalBelow;

          if (!isCrossingInk && cur <= surround - 4) {
            out[rowOffset + x] = Math.max(cur, surround, paper);
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
  const edgeMarginX = Math.max(2, Math.round(w * 0.004));
  const edgeMarginY = Math.max(2, Math.round(h * 0.004));

  interface KeptComp {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
    cw: number;
    ch: number;
    area: number;
    pixels: number[];
    rejected: boolean;
  }

  const keptComps: KeptComp[] = [];

  for (let startIdx = 0; startIdx < totalPixels; startIdx++) {
    if (visited[startIdx] || binary[startIdx] !== 0) continue;

    visited[startIdx] = 1;
    queue.length = 0;
    const compPixels: number[] = [];
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
    if (area > totalPixels * 0.025) continue;
    if (cw > w * 0.65 && ch > h * 0.22) continue;
    if (ch > h * 0.42) continue;

    // Reject solid dark horizontal footer/header bars touching top/bottom edges
    if (cw > w * 0.5 && (minY <= edgeMarginY || maxY >= h - 1 - edgeMarginY)) {
      continue;
    }

    // Reject large edge-touching clutter (hands holding paper, desk edges, spiral spines)
    if (touchesEdge && (ch > h * 0.18 || cw > w * 0.45 || area > totalPixels * 0.012)) {
      continue;
    }

    // Reject long thin horizontal/vertical border rules or spiral spines
    if (cw > w * 0.65 && ch <= 5) continue;
    if (ch > h * 0.45 && cw <= 6) continue;

    keptComps.push({
      minX,
      maxX,
      minY,
      maxY,
      cw,
      ch,
      area,
      pixels: compPixels,
      rejected: false,
    });
  }

  // Strip spiral-binding coils along left & right vertical margins
  const stripMarginSpiralCoils = (side: 'left' | 'right') => {
    const candidates = keptComps.filter((c) => {
      if (c.rejected) return false;
      if (c.ch < 4 || c.ch > h * 0.12 || c.cw > w * 0.12) return false;

      if (side === 'left') {
        if (c.maxX >= w * 0.14) return false;
        // Verify no immediate horizontal text continuation to the right on the same row band
        const hasRightNeighbor = keptComps.some(
          (other) =>
            other !== c &&
            !other.rejected &&
            other.minX > c.maxX &&
            other.minX <= c.maxX + Math.max(10, Math.round(c.ch * 0.85)) &&
            other.maxY >= c.minY - 4 &&
            other.minY <= c.maxY + 4
        );
        return !hasRightNeighbor;
      } else {
        if (c.minX <= w * 0.86) return false;
        const hasLeftNeighbor = keptComps.some(
          (other) =>
            other !== c &&
            !other.rejected &&
            other.maxX < c.minX &&
            other.maxX >= c.minX - Math.max(10, Math.round(c.ch * 0.85)) &&
            other.maxY >= c.minY - 4 &&
            other.minY <= c.maxY + 4
        );
        return !hasLeftNeighbor;
      }
    });

    if (candidates.length === 0) return;

    // Group vertically sorted candidates that align along the margin strip
    const sorted = [...candidates].sort((a, b) => (a.minY + a.maxY) - (b.minY + b.maxY));

    // Check if edge-touching margin components exist (>= 2 touching extreme edge)
    const edgeTouching = sorted.filter((c) =>
      side === 'left' ? c.minX <= Math.max(edgeMarginX, Math.round(w * 0.015)) : c.maxX >= w - 1 - Math.max(edgeMarginX, Math.round(w * 0.015))
    );
    if (edgeTouching.length >= 2) {
      for (const c of edgeTouching) {
        c.rejected = true;
      }
    }

    // Detect periodic repeating coils in vertical margin strip (x < 0.14*w or x > 0.86*w)
    const aligned = sorted.filter((c) =>
      side === 'left' ? c.minX <= w * 0.06 || c.maxX < w * 0.14 : c.maxX >= w * 0.94 || c.minX > w * 0.86
    );
    if (aligned.length >= 3) {
      // Cluster by similar X-center so random words at varying indents don't trigger
      const xCenters = aligned.map((c) => (c.minX + c.maxX) * 0.5).sort((a, b) => a - b);
      const medianX = xCenters[Math.floor(xCenters.length * 0.5)];
      const colAligned = aligned.filter(
        (c) => Math.abs((c.minX + c.maxX) * 0.5 - medianX) <= Math.max(10, w * 0.045)
      );

      if (colAligned.length >= 3) {
        const yCenters = colAligned.map((c) => (c.minY + c.maxY) * 0.5);
        const gaps: number[] = [];
        for (let i = 1; i < yCenters.length; i++) {
          const g = yCenters[i] - yCenters[i - 1];
          if (g >= 6) gaps.push(g);
        }
        if (gaps.length >= 2) {
          const sortedGaps = [...gaps].sort((a, b) => a - b);
          const medianGap = sortedGaps[Math.floor(sortedGaps.length * 0.5)];
          const consistentGaps = gaps.filter(
            (g) => Math.abs(g - medianGap) <= Math.max(4, medianGap * 0.42)
          ).length;
          if (consistentGaps >= Math.max(2, Math.floor(gaps.length * 0.6))) {
            for (const c of colAligned) {
              c.rejected = true;
            }
          }
        }
      }
    }
  };

  if (w >= 160 && h >= 120) {
    stripMarginSpiralCoils('left');
    stripMarginSpiralCoils('right');
  }

  for (const comp of keptComps) {
    if (comp.rejected) continue;
    for (let i = 0; i < comp.pixels.length; i++) {
      out[comp.pixels[i]] = 0;
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

  // Measure per-column and per-row paper coverage (>= 135 handles indoor shaded paper cleanly)
  const colBrightRatio = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    let bright = 0;
    for (let y = 0; y < h; y += 2) {
      if (gray[y * w + x] >= 135) bright++;
    }
    colBrightRatio[x] = bright / Math.ceil(h / 2);
  }

  const rowBrightRatio = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    let bright = 0;
    const rowOff = y * w;
    for (let x = 0; x < w; x += 2) {
      if (gray[rowOff + x] >= 135) bright++;
    }
    rowBrightRatio[y] = bright / Math.ceil(w / 2);
  }

  // Trim ONLY dark outer margins inward from the frame edges so vertical ink columns
  // or dark notebook margin rules inside the page can NEVER split/erase the text region.
  let x0 = 0;
  let x1 = w - 1;
  while (x0 < w * 0.28 && colBrightRatio[x0] < 0.35) x0++;
  while (x1 > w * 0.72 && colBrightRatio[x1] < 0.35) x1--;

  let y0 = 0;
  let y1 = h - 1;
  while (y0 < h * 0.28 && rowBrightRatio[y0] < 0.35) y0++;
  while (y1 > h * 0.72 && rowBrightRatio[y1] < 0.35) y1--;

  const cropW = x1 - x0 + 1;
  const cropH = y1 - y0 + 1;
  const changed = cropW < w * 0.96 || cropH < h * 0.96;

  if (!changed || cropW < Math.max(60, w * 0.42) || cropH < Math.max(40, h * 0.42)) {
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
