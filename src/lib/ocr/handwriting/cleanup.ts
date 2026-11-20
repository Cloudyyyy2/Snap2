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
  type SlantOptions,
  type SlantResult,
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
  estimateSlantAngle,
  estimateCursiveSlantAngle,
  shearDeslantGrayscale,
  shearGrayscale,
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



interface CleanupComp {
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

function extractBinaryComponents(
  binary: Uint8ClampedArray,
  w: number,
  h: number
): CleanupComp[] {
  const totalPixels = w * h;
  const visited = new Uint8Array(totalPixels);
  const queue: number[] = [];
  const comps: CleanupComp[] = [];

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

    comps.push({
      minX,
      maxX,
      minY,
      maxY,
      cw: maxX - minX + 1,
      ch: maxY - minY + 1,
      area: compPixels.length,
      pixels: compPixels,
      rejected: false,
    });
  }

  return comps;
}

function rejectSpiralAndCornerInComps(
  comps: CleanupComp[],
  w: number,
  h: number
): void {
  if (w < 160 || h < 120 || comps.length === 0) return;

  // (a) Rounded/dark corner wedges and thin corner-touching border bars
  for (const c of comps) {
    if (c.rejected) continue;
    const touchesH = c.minY <= 2 || c.maxY >= h - 3;
    const touchesV = c.minX <= 2 || c.maxX >= w - 3;
    const isCornerWedge =
      touchesH &&
      touchesV &&
      c.cw <= w * 0.2 &&
      c.ch <= h * 0.2 &&
      (c.area / Math.max(1, c.cw * c.ch) >= 0.38 || c.area <= 300);
    const isCornerBorderBar =
      touchesH &&
      touchesV &&
      ((c.ch <= Math.max(6, Math.round(h * 0.02)) && c.cw <= w * 0.35) ||
        (c.cw <= Math.max(6, Math.round(w * 0.025)) && c.ch <= h * 0.35));
    if (isCornerWedge || isCornerBorderBar) {
      c.rejected = true;
    }
  }

  // (b) Left & Right spiral-binding coil holes and edge-touching margin stubs
  const stripMarginSpiralCoils = (side: 'left' | 'right') => {
    const candidates = comps.filter((c) => {
      if (c.rejected) return false;
      if (c.ch < 4 || c.cw > w * 0.14) return false;
      const touchesExtreme = side === 'left' ? c.minX <= 2 : c.maxX >= w - 3;
      const maxH = touchesExtreme ? h * 0.22 : h * 0.12;
      if (c.ch > maxH) return false;

      if (side === 'left') {
        if (c.maxX >= w * 0.14) return false;
        const hasRightNeighbor = comps.some(
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
        const hasLeftNeighbor = comps.some(
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

    const sorted = [...candidates].sort((a, b) => (a.minY + a.maxY) - (b.minY + b.maxY));

    // Only treat as frame-edge spiral stubs when they truly touch/clip the outer border
    const edgeTouching = sorted.filter((c) =>
      side === 'left' ? c.minX <= 2 : c.maxX >= w - 3
    );
    if (edgeTouching.length >= 2) {
      for (const c of edgeTouching) {
        c.rejected = true;
      }
    }

    const aligned = sorted.filter((c) => {
      if (c.rejected) return false;
      // Require coil/hole ring morphology (avoid stripping small '-' / '•' bullets or thin marks)
      if (c.ch < 7 || c.cw < 5 || c.area < 30) return false;
      return side === 'left'
        ? c.minX <= w * 0.06 || c.maxX < w * 0.14
        : c.maxX >= w * 0.94 || c.minX > w * 0.86;
    });
    if (aligned.length >= 3) {
      const xCenters = aligned.map((c) => (c.minX + c.maxX) * 0.5).sort((a, b) => a - b);
      const medianX = xCenters[Math.floor(xCenters.length * 0.5)];
      const colAligned = aligned.filter(
        (c) => Math.abs((c.minX + c.maxX) * 0.5 - medianX) <= Math.max(10, w * 0.045)
      );

      if (colAligned.length >= 3) {
        // Distinguish real inset spiral coils from bulleted/numbered lists or aligned right prices:
        // if >= 65% sit directly alongside same-row interior text, preserve them as list/price tokens.
        const colSet = new Set(colAligned);
        const alignedWithTextRows = colAligned.filter((c) =>
          comps.some(
            (other) =>
              !other.rejected &&
              !colSet.has(other) &&
              other.area >= 12 &&
              other.maxY >= c.minY - 4 &&
              other.minY <= c.maxY + 4 &&
              (side === 'left' ? other.minX > c.maxX + 6 : other.maxX < c.minX - 6)
          )
        ).length;
        const mostlyTouchesOuterEdge =
          colAligned.filter((c) => (side === 'left' ? c.minX <= 3 : c.maxX >= w - 4)).length >=
          Math.ceil(colAligned.length * 0.6);

        if (mostlyTouchesOuterEdge || alignedWithTextRows < colAligned.length * 0.65) {
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
    }
  };

  stripMarginSpiralCoils('left');
  stripMarginSpiralCoils('right');

  // (c) Edge-clipped partial-column corner bleeds (e.g. clipped next-column words on right/left edge)
  const chasm = Math.max(45, Math.round(w * 0.22));
  const clusterGap = Math.max(14, Math.round(h * 0.05));

  const stripEdgeClippedClusters = (side: 'left' | 'right') => {
    const stripComps = comps.filter((c) => {
      if (c.rejected) return false;
      if (c.cw > w * 0.16 || c.ch > h * 0.16) return false;
      return side === 'left' ? c.maxX <= w * 0.18 : c.minX >= w * 0.82;
    });
    if (stripComps.length === 0) return;

    const visitedStrip = new Uint8Array(stripComps.length);
    const qualifyingClusters: Array<{
      cluster: CleanupComp[];
      minY: number;
      maxY: number;
      hasRowInterior: boolean;
    }> = [];

    for (let i = 0; i < stripComps.length; i++) {
      if (visitedStrip[i]) continue;
      visitedStrip[i] = 1;
      const cluster: CleanupComp[] = [stripComps[i]];
      const queueIdx: number[] = [i];
      let head = 0;

      while (head < queueIdx.length) {
        const cur = stripComps[queueIdx[head++]];
        for (let j = 0; j < stripComps.length; j++) {
          if (visitedStrip[j]) continue;
          const other = stripComps[j];
          const gapX = Math.max(0, cur.minX - other.maxX, other.minX - cur.maxX);
          const gapY = Math.max(0, cur.minY - other.maxY, other.minY - cur.maxY);
          if (gapX <= clusterGap && gapY <= clusterGap) {
            visitedStrip[j] = 1;
            queueIdx.push(j);
            cluster.push(other);
          }
        }
      }

      let clMinX = w;
      let clMaxX = -1;
      let clMinY = h;
      let clMaxY = -1;
      let touchesBorder = false;

      for (const c of cluster) {
        if (c.minX < clMinX) clMinX = c.minX;
        if (c.maxX > clMaxX) clMaxX = c.maxX;
        if (c.minY < clMinY) clMinY = c.minY;
        if (c.maxY > clMaxY) clMaxY = c.maxY;
        // Require genuine outer frame-border clipping (within 2px of left/right frame edge)
        if (side === 'left' ? c.minX <= 1 : c.maxX >= w - 2) {
          touchesBorder = true;
        }
      }

      const clW = clMaxX - clMinX + 1;
      const clH = clMaxY - clMinY + 1;
      if (!touchesBorder || clW > w * 0.16 || clH > h * 0.28) continue;

      const clusterSet = new Set(cluster);
      let hasCloseNeighbor = false;
      let bandInteriorInk = 0;
      let pageInteriorInk = 0;

      for (const other of comps) {
        if (other.rejected || clusterSet.has(other)) continue;
        const inInterior = other.minX < w * 0.78 && other.maxX > w * 0.22;
        if (inInterior) pageInteriorInk += other.area;

        const overlapsBand = other.maxY >= clMinY - 6 && other.minY <= clMaxY + 6;
        if (!overlapsBand) continue;

        if (side === 'right') {
          if (other.maxX >= clMinX - chasm && other.minX < clMinX) {
            hasCloseNeighbor = true;
            break;
          }
          if (other.maxX < clMinX - chasm) {
            bandInteriorInk += other.area;
          }
        } else {
          if (other.minX <= clMaxX + chasm && other.maxX > clMaxX) {
            hasCloseNeighbor = true;
            break;
          }
          if (other.minX > clMaxX + chasm) {
            bandInteriorInk += other.area;
          }
        }
      }

      if (!hasCloseNeighbor && (bandInteriorInk >= 25 || pageInteriorInk >= 60)) {
        qualifyingClusters.push({
          cluster,
          minY: clMinY,
          maxY: clMaxY,
          hasRowInterior: bandInteriorInk >= 25,
        });
      }
    }

    if (qualifyingClusters.length === 0) return;

    // If multiple vertically separated row-aligned clusters appear across the page (e.g., right-aligned
    // prices or a narrow right column), preserve them; only strip isolated corner/bleed clusters.
    if (qualifyingClusters.length >= 2) {
      const minYAll = Math.min(...qualifyingClusters.map((q) => q.minY));
      const maxYAll = Math.max(...qualifyingClusters.map((q) => q.maxY));
      const rowAlignedCount = qualifyingClusters.filter((q) => q.hasRowInterior).length;
      if (rowAlignedCount >= 2 && maxYAll - minYAll >= h * 0.22) {
        return;
      }
    }

    for (const q of qualifyingClusters) {
      for (const c of q.cluster) {
        c.rejected = true;
      }
    }
  };

  stripEdgeClippedClusters('left');
  stripEdgeClippedClusters('right');
}

/**
 * Strips left/right margin spiral holes, rounded corner wedges, and edge-clipped
 * partial-column corner bleeds from a binary buffer (0 = ink, 255 = paper) without
 * disturbing interior text or prices. Safely no-ops on frames smaller than 160x120.
 */
export function stripSpiralAndCornerArtifacts(
  binary: Uint8ClampedArray,
  w: number,
  h: number
): Uint8ClampedArray {
  if (w < 160 || h < 120 || binary.length < w * h) {
    return new Uint8ClampedArray(binary);
  }

  const out = new Uint8ClampedArray(w * h).fill(255);
  const comps = extractBinaryComponents(binary, w, h);
  rejectSpiralAndCornerInComps(comps, w, h);

  for (const comp of comps) {
    if (comp.rejected) continue;
    for (let i = 0; i < comp.pixels.length; i++) {
      out[comp.pixels[i]] = 0;
    }
  }

  return out;
}

/**
 * Filters out non-text background clutter (desks, room shadows, fingers, spiral bindings,
 * corner wedges, edge-clipped bleeds, long dividers, and 1-2px speckles) from a Sauvola
 * binary buffer (0 = ink, 255 = paper).
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
  const edgeMarginX = Math.max(2, Math.round(w * 0.004));
  const edgeMarginY = Math.max(2, Math.round(h * 0.004));

  const allComps = extractBinaryComponents(binary, w, h);
  const keptComps: CleanupComp[] = [];
  const smallDotCandidates: CleanupComp[] = [];

  for (const comp of allComps) {
    const { minX, maxX, minY, maxY, cw, ch, area } = comp;
    const touchesEdge =
      minX <= edgeMarginX ||
      maxX >= w - 1 - edgeMarginX ||
      minY <= edgeMarginY ||
      maxY >= h - 1 - edgeMarginY;

    // Reject tiny noise specks, but defer 2x2 punctuation/period dots near glyphs
    if (area < 4) continue;
    if (area === 4 && cw <= 2 && ch <= 2) {
      smallDotCandidates.push(comp);
      continue;
    }
    if (cw <= 2 && ch <= 2) continue;

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

    keptComps.push(comp);
  }

  // Keep 2x2 period/punctuation dots that sit directly beside a real glyph component
  for (const dot of smallDotCandidates) {
    const nearGlyph = keptComps.some(
      (g) =>
        g.area >= 12 &&
        g.ch >= 6 &&
        dot.maxY >= g.minY - 3 &&
        dot.minY <= g.maxY + 3 &&
        dot.minX <= g.maxX + 6 &&
        dot.maxX >= g.minX - 6
    );
    if (nearGlyph) {
      keptComps.push(dot);
    }
  }

  rejectSpiralAndCornerInComps(keptComps, w, h);

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
