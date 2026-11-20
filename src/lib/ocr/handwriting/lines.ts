import { filterTextComponents, stripSpiralAndCornerArtifacts } from './cleanup';

export interface LineBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface CroppedLine {
  data: Uint8ClampedArray;
  width: number;
  height: number;
  box: LineBox;
}

/**
 * Segments a binarized document (0 = ink, 255 = paper) into individual upright text lines.
 * Filters obvious non-text clutter on camera-sized frames, uses smoothed horizontal ink
 * projection splitting touching ascenders/descenders at projection valleys, and trims
 * detached border smudges horizontally.
 */
export function segmentLines(
  binary: Uint8ClampedArray,
  w: number,
  h: number,
  options?: { skipFilter?: boolean }
): LineBox[] {
  if (w === 0 || h === 0) return [];

  let activeBinary = binary;
  if (!options?.skipFilter) {
    if (w >= 280 && h >= 200) {
      activeBinary = filterTextComponents(binary, w, h);
    } else if (w >= 160 && h >= 120) {
      activeBinary = stripSpiralAndCornerArtifacts(binary, w, h);
    }
  }

  const proj = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    let ink = 0;
    const rowOffset = y * w;
    for (let x = 0; x < w; x++) {
      if (activeBinary[rowOffset + x] === 0) ink++;
    }
    proj[y] = ink;
  }

  // Smooth horizontal projection (radius 2)
  const smooth = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    let sum = 0;
    let cnt = 0;
    for (let dy = -2; dy <= 2; dy++) {
      const ny = y + dy;
      if (ny >= 0 && ny < h) {
        sum += proj[ny];
        cnt++;
      }
    }
    smooth[y] = sum / cnt;
  }

  let maxProj = 0;
  for (let y = 0; y < h; y++) {
    if (smooth[y] > maxProj) maxProj = smooth[y];
  }
  if (maxProj < 2) return [];

  // Identify local peaks corresponding to line centers
  const peakThreshold = Math.max(2, maxProj * 0.14);
  const minPeakDistance = Math.max(10, Math.round(h * 0.025));

  const rawPeaks: number[] = [];
  for (let y = 1; y < h - 1; y++) {
    if (
      smooth[y] >= peakThreshold &&
      smooth[y] >= smooth[y - 1] &&
      smooth[y] >= smooth[y + 1]
    ) {
      rawPeaks.push(y);
    }
  }

  // Merge peaks that belong to the same text line (closer than minPeakDistance without deep valley)
  const peaks: number[] = [];
  for (const p of rawPeaks) {
    if (peaks.length === 0) {
      peaks.push(p);
      continue;
    }
    const prev = peaks[peaks.length - 1];
    let minBetween = Infinity;
    for (let y = prev; y <= p; y++) {
      if (smooth[y] < minBetween) minBetween = smooth[y];
    }
    const weakerPeak = Math.min(smooth[prev], smooth[p]);
    const valleyRatio =
      p - prev >= Math.max(18, Math.round(minPeakDistance * 1.8)) ? 0.48 : 0.35;
    const hasDeepValley = minBetween <= weakerPeak * valleyRatio;

    if (!hasDeepValley) {
      if (smooth[p] > smooth[prev]) {
        peaks[peaks.length - 1] = p;
      }
    } else {
      peaks.push(p);
    }
  }

  if (peaks.length === 0) return [];

  // Determine vertical boundaries between peaks by finding the midpoint of the lowest valley
  // between adjacent peaks so ascender/descender margins never bleed across neighboring lines.
  const lineRanges: Array<{ y0: number; y1: number }> = [];
  const activeThreshold = Math.max(0.5, maxProj * 0.015);

  const valleys: number[] = [];
  for (let i = 0; i < peaks.length - 1; i++) {
    const pA = peaks[i];
    const pB = peaks[i + 1];
    let minVal = Infinity;
    for (let y = pA + 1; y < pB; y++) {
      if (smooth[y] < minVal) minVal = smooth[y];
    }
    let firstMin = -1;
    let lastMin = -1;
    for (let y = pA + 1; y < pB; y++) {
      if (smooth[y] <= minVal + 0.5) {
        if (firstMin === -1) firstMin = y;
        lastMin = y;
      }
    }
    const valleyMid =
      firstMin !== -1 ? Math.floor((firstMin + lastMin) / 2) : Math.floor((pA + pB) / 2);
    valleys.push(Math.max(pA, Math.min(pB - 1, valleyMid)));
  }

  for (let i = 0; i < peaks.length; i++) {
    const upperLimit = i === 0 ? 0 : valleys[i - 1] + 1;
    const lowerLimit = i === peaks.length - 1 ? h - 1 : valleys[i];

    let y0 = peaks[i];
    while (y0 > upperLimit && smooth[y0 - 1] >= activeThreshold) y0--;

    let y1 = peaks[i];
    while (y1 < lowerLimit && smooth[y1 + 1] >= activeThreshold) y1++;

    // Preserve thin 1-2px cursive ascender/descender tips (f, g, j, l, p, q, y) within local pitch
    // and expand gently by up to 0.32 * local line pitch, strictly capped at the valley midpoint.
    const prevDist = i > 0 ? peaks[i] - peaks[i - 1] : Infinity;
    const nextDist = i < peaks.length - 1 ? peaks[i + 1] - peaks[i] : Infinity;
    const localPitch = Number.isFinite(Math.min(prevDist, nextDist))
      ? Math.min(prevDist, nextDist)
      : Math.max(20, Math.round((y1 - y0 + 1) * 1.5));
    const maxRawReach = Math.round(0.45 * localPitch);
    const rawTopLimit = Math.max(upperLimit, peaks[i] - maxRawReach);
    const rawBotLimit = Math.min(lowerLimit, peaks[i] + maxRawReach);
    while (y0 > rawTopLimit && (smooth[y0 - 1] >= activeThreshold || proj[y0 - 1] > 0)) y0--;
    while (y1 < rawBotLimit && (smooth[y1 + 1] >= activeThreshold || proj[y1 + 1] > 0)) y1++;

    const maxAscDescPad = Math.max(4, Math.round(0.32 * localPitch));

    y0 = Math.max(upperLimit, y0 - maxAscDescPad);
    y1 = Math.min(lowerLimit, y1 + maxAscDescPad);

    lineRanges.push({ y0, y1 });
  }

  // Refine x0..x1 using vertical column density to drop distant edge smudges and isolated corner bleeds
  const colInk = new Int32Array(w);
  const largeGap = Math.max(45, Math.round(w * 0.14));
  const edgeChasm = Math.max(45, Math.round(w * 0.22));

  type LineSeg = { x0: number; x1: number; ink: number };
  const rangeEntries: Array<{
    rangeIdx: number;
    range: { y0: number; y1: number };
    segments: LineSeg[];
    significant: LineSeg[];
    hasLeftEdgeSeg: boolean;
    hasRightEdgeSeg: boolean;
  }> = [];

  let leftEdgeLineCount = 0;
  let rightEdgeLineCount = 0;

  for (let rangeIdx = 0; rangeIdx < lineRanges.length; rangeIdx++) {
    const range = lineRanges[rangeIdx];
    colInk.fill(0);
    let totalInk = 0;

    for (let y = range.y0; y <= range.y1; y++) {
      const rowOffset = y * w;
      for (let x = 0; x < w; x++) {
        if (activeBinary[rowOffset + x] === 0) {
          colInk[x]++;
          totalInk++;
        }
      }
    }

    if (totalInk < 12) continue;

    // Group horizontal runs of ink separated by wide whitespace gaps
    const segments: LineSeg[] = [];
    let segStart = -1;
    let segLast = -1;
    let segInk = 0;

    for (let x = 0; x < w; x++) {
      if (colInk[x] > 0) {
        if (segStart === -1) {
          segStart = x;
          segInk = colInk[x];
        } else if (x - segLast > largeGap) {
          segments.push({ x0: segStart, x1: segLast, ink: segInk });
          segStart = x;
          segInk = colInk[x];
        } else {
          segInk += colInk[x];
        }
        segLast = x;
      }
    }
    if (segStart !== -1) {
      segments.push({ x0: segStart, x1: segLast, ink: segInk });
    }

    const maxSegInk = segments.reduce((m, s) => Math.max(m, s.ink), 0);
    const significant = segments.filter(
      (s, sIdx) =>
        s.ink >= Math.max(12, totalInk * 0.08) ||
        segments.length === 1 ||
        // Preserve leading left-margin list bullets / item numbers separated by a tab gap
        (sIdx === 0 && s.x0 >= 3 && s.x1 < w * 0.18 && s.ink >= 15)
    );

    let hasLeftEdgeSeg = false;
    let hasRightEdgeSeg = false;
    if (significant.length >= 2) {
      const first = significant[0];
      const last = significant[significant.length - 1];
      if (
        first.x0 <= 1 &&
        first.x1 < w * 0.16 &&
        significant[1].x0 - first.x1 >= edgeChasm &&
        first.ink < maxSegInk * 0.65
      ) {
        hasLeftEdgeSeg = true;
        leftEdgeLineCount++;
      }
      if (
        last.x1 >= w - 2 &&
        last.x0 > w * 0.84 &&
        last.x0 - significant[significant.length - 2].x1 >= edgeChasm &&
        last.ink < maxSegInk * 0.65
      ) {
        hasRightEdgeSeg = true;
        rightEdgeLineCount++;
      }
    }

    rangeEntries.push({
      rangeIdx,
      range,
      segments,
      significant,
      hasLeftEdgeSeg,
      hasRightEdgeSeg,
    });
  }

  const boxes: LineBox[] = [];
  for (const entry of rangeEntries) {
    const { rangeIdx, range, segments } = entry;
    let significant = entry.significant;

    // Drop border-clipped corner bleeds ONLY when isolated to a top/bottom corner (not multi-row columns/prices)
    const inCornerBand = range.y0 <= h * 0.22 || range.y1 >= h * 0.78;
    if (significant.length >= 2 && inCornerBand) {
      const maxSegInk = segments.reduce((m, s) => Math.max(m, s.ink), 0);
      const withoutEdgeBleeds = significant.filter((s, idx) => {
        const isLeftEdgeBleed =
          leftEdgeLineCount < 2 &&
          s.x0 <= 1 &&
          s.x1 < w * 0.16 &&
          idx + 1 < significant.length &&
          significant[idx + 1].x0 - s.x1 >= edgeChasm &&
          s.ink < maxSegInk * 0.65;
        const isRightEdgeBleed =
          rightEdgeLineCount < 2 &&
          s.x1 >= w - 2 &&
          s.x0 > w * 0.84 &&
          idx > 0 &&
          s.x0 - significant[idx - 1].x1 >= edgeChasm &&
          s.ink < maxSegInk * 0.65;
        return !isLeftEdgeBleed && !isRightEdgeBleed;
      });
      if (withoutEdgeBleeds.length > 0) {
        significant = withoutEdgeBleeds;
      }
    }
    const chosen = significant.length > 0 ? significant : segments;
    if (chosen.length === 0) continue;

    const boundX0 = chosen[0].x0;
    const boundX1 = chosen[chosen.length - 1].x1;

    let minX = boundX1;
    let maxX = boundX0;
    let minY = range.y1;
    let maxY = range.y0;
    let inkPixels = 0;

    for (let y = range.y0; y <= range.y1; y++) {
      const rowOffset = y * w;
      for (let x = boundX0; x <= boundX1; x++) {
        if (activeBinary[rowOffset + x] === 0) {
          inkPixels++;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    const linePitch =
      peaks.length > 1
        ? (peaks[peaks.length - 1] - peaks[0]) / (peaks.length - 1)
        : Math.max(30, range.y1 - range.y0 + 1);
    const margin = Math.max(12, Math.round(linePitch * 0.32));
    let safeY0 = Math.max(0, minY - margin);
    let safeY1 = Math.min(h - 1, maxY + margin);

    // Cap padding at the valley boundary between adjacent lines without off-by-one ink clipping
    if (rangeIdx > 0) {
      safeY0 = Math.min(minY, Math.max(range.y0, safeY0));
    }
    if (rangeIdx < lineRanges.length - 1) {
      safeY1 = Math.max(maxY, Math.min(range.y1, safeY1));
    }

    const boxH = safeY1 - safeY0 + 1;
    const boxW = maxX - minX + 1;
    const minBoxW = w >= 280 ? 24 : 8;
    const minBoxH = h >= 200 ? 12 : 6;
    const minInk = w >= 280 ? 40 : 12;
    if (maxX >= minX && boxH >= minBoxH && boxW >= minBoxW && inkPixels >= minInk) {
      boxes.push({ x0: minX, y0: safeY0, x1: maxX, y1: safeY1 });
    }
  }

  return boxes;
}

/**
 * Splits overly wide line boxes (aspect > maxAspect) at deep inter-word whitespace gaps
 * or stem-safe minimum-ink ligature valleys so downstream vision encoders do not squash characters
 * and stems are never sliced mid-glyph.
 */
export function splitWideLineBox(
  binary: Uint8ClampedArray,
  w: number,
  h: number,
  box: LineBox,
  maxAspect = 6.5
): LineBox[] {
  const boxW = box.x1 - box.x0 + 1;
  const boxH = Math.max(1, box.y1 - box.y0 + 1);
  if (boxW / boxH <= maxAspect || boxW < 180) {
    return [box];
  }

  const maxChunkW = Math.max(120, Math.round(boxH * maxAspect));
  const minSplitGap = Math.max(6, Math.round(boxH * 0.22));
  const colInk = new Int32Array(boxW);
  const colMinY = new Int32Array(boxW).fill(boxH);
  const colMaxY = new Int32Array(boxW).fill(-1);
  const colSpan = new Int32Array(boxW);

  const yStart = Math.max(0, box.y0);
  const yEnd = Math.min(h - 1, box.y1);
  let minInkY = boxH;
  let maxInkY = -1;
  for (let y = yStart; y <= yEnd; y++) {
    const ly = y - box.y0;
    const rowOffset = y * w;
    for (let x = box.x0; x <= box.x1; x++) {
      if (x < 0 || x >= w) continue;
      if (binary[rowOffset + x] === 0) {
        const lx = x - box.x0;
        colInk[lx]++;
        if (ly < colMinY[lx]) colMinY[lx] = ly;
        if (ly > colMaxY[lx]) colMaxY[lx] = ly;
        if (ly < minInkY) minInkY = ly;
        if (ly > maxInkY) maxInkY = ly;
      }
    }
  }
  const nonZeroSpans: number[] = [];
  for (let lx = 0; lx < boxW; lx++) {
    if (colInk[lx] > 0 && colMaxY[lx] >= colMinY[lx]) {
      colSpan[lx] = colMaxY[lx] - colMinY[lx] + 1;
      nonZeroSpans.push(colSpan[lx]);
    }
  }
  const rawInkHeight = maxInkY >= minInkY ? maxInkY - minInkY + 1 : boxH;
  nonZeroSpans.sort((a, b) => a - b);
  const p85Span =
    nonZeroSpans.length > 0
      ? nonZeroSpans[Math.floor(nonZeroSpans.length * 0.85)]
      : rawInkHeight;
  const inkHeight = Math.max(
    8,
    Math.min(rawInkHeight, Math.round(p85Span * 1.15))
  );

  const smoothColInk = new Float32Array(boxW);
  for (let lx = 0; lx < boxW; lx++) {
    let sum = 0;
    let cnt = 0;
    for (let dx = -1; dx <= 1; dx++) {
      const nx = lx + dx;
      if (nx >= 0 && nx < boxW) {
        sum += colInk[nx];
        cnt++;
      }
    }
    smoothColInk[lx] = sum / cnt;
  }

  const findCleanGap = (
    fromX: number,
    toX: number,
    minGapW: number,
    extendLimit: number
  ): number => {
    let bestCenter = -1;
    let bestWidth = 0;
    let gapStart = -1;

    for (let lx = fromX; lx <= toX; lx++) {
      if (colInk[lx] <= 1) {
        if (gapStart === -1) gapStart = lx;
      } else if (gapStart !== -1) {
        const gw = lx - gapStart;
        if (gw >= minGapW && gw >= bestWidth) {
          bestWidth = gw;
          bestCenter = Math.floor((gapStart + lx - 1) / 2);
        }
        gapStart = -1;
      }
    }
    if (gapStart !== -1) {
      let gapEnd = toX;
      while (gapEnd + 1 <= extendLimit && colInk[gapEnd + 1] <= 1) {
        gapEnd++;
      }
      const gw = gapEnd - gapStart + 1;
      if (gw >= minGapW && gw >= bestWidth) {
        bestWidth = gw;
        bestCenter = Math.floor((gapStart + gapEnd) / 2);
      }
    }
    return bestCenter;
  };

  const maxLigatureInk = Math.max(3, Math.round(inkHeight * 0.25));
  const maxLigatureSpan = inkHeight * 0.45;
  const stemSpanThreshold = inkHeight * 0.55;

  const isSafeLigatureCol = (lx: number): boolean => {
    if (colInk[lx] > maxLigatureInk || colSpan[lx] > maxLigatureSpan) return false;
    // Allow genuine 1-2px minimum-ink ligature bridges even when +/-1px touches adjacent tall stems
    if (colInk[lx] > 0 && colInk[lx] <= 3 && colSpan[lx] <= inkHeight * 0.3) {
      return true;
    }
    for (let dx = -1; dx <= 1; dx++) {
      const nx = lx + dx;
      if (nx >= 0 && nx < boxW && colSpan[nx] >= stemSpanThreshold) {
        return false;
      }
    }
    return true;
  };

  const findSafeLigatureValley = (fromX: number, toX: number): number => {
    let bestScore = Infinity;
    let bestRuns: number[] = [];

    for (let lx = fromX; lx <= toX; lx++) {
      if (!isSafeLigatureCol(lx)) continue;
      const score = smoothColInk[lx];
      if (score < bestScore - 1e-6) {
        bestScore = score;
        bestRuns = [lx];
      } else if (Math.abs(score - bestScore) <= 1e-6) {
        bestRuns.push(lx);
      }
    }

    if (bestRuns.length === 0) return -1;
    // Pick midpoint of the widest contiguous run among minimum-score columns
    let bestStart = bestRuns[0];
    let bestLen = 1;
    let curStart = bestRuns[0];
    let curLen = 1;
    for (let i = 1; i < bestRuns.length; i++) {
      if (bestRuns[i] === bestRuns[i - 1] + 1) {
        curLen++;
        if (curLen > bestLen) {
          bestLen = curLen;
          bestStart = curStart;
        }
      } else {
        curStart = bestRuns[i];
        curLen = 1;
      }
    }
    return bestStart + Math.floor((bestLen - 1) / 2);
  };

  const chunks: LineBox[] = [];
  let cursor = 0;

  while (cursor < boxW) {
    if (boxW - cursor <= maxChunkW) {
      chunks.push({
        x0: box.x0 + cursor,
        y0: box.y0,
        x1: box.x1,
        y1: box.y1,
      });
      break;
    }

    const searchStart = Math.min(boxW - 1, cursor + Math.round(maxChunkW * 0.55));
    const lookaheadEnd = Math.min(boxW - 1, cursor + Math.round(maxChunkW * 1.25));
    const extendEnd = Math.min(boxW - 1, cursor + Math.round(maxChunkW * 1.45));

    // Tier 1: clean inter-word gap >= minSplitGap
    let splitAt = findCleanGap(searchStart, lookaheadEnd, minSplitGap, extendEnd);

    // Tier 2: narrower clean whitespace gap >= 2px
    if (splitAt === -1) {
      splitAt = findCleanGap(searchStart, lookaheadEnd, 2, extendEnd);
    }

    // Tier 3: stem-safe minimum-ink ligature valley
    if (splitAt === -1) {
      splitAt = findSafeLigatureValley(searchStart, lookaheadEnd);
    }

    // Tier 4: scan further forward for next safe gap/ligature; otherwise abort split to avoid slicing a glyph
    if (splitAt === -1 && lookaheadEnd + 1 < boxW - 1) {
      splitAt = findCleanGap(lookaheadEnd + 1, boxW - 2, 2, boxW - 1);
      if (splitAt === -1) {
        splitAt = findSafeLigatureValley(lookaheadEnd + 1, boxW - 2);
      }
    }

    if (splitAt === -1 || splitAt <= cursor || splitAt >= boxW - 1) {
      chunks.push({
        x0: box.x0 + cursor,
        y0: box.y0,
        x1: box.x1,
        y1: box.y1,
      });
      break;
    }

    chunks.push({
      x0: box.x0 + cursor,
      y0: box.y0,
      x1: box.x0 + splitAt,
      y1: box.y1,
    });
    cursor = splitAt + 1;
    while (cursor < boxW && colInk[cursor] === 0) cursor++;
  }

  return chunks;
}

export interface WordBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  lineIndex: number;
  wordIndex: number;
  squareBox: {
    x0: number;
    y0: number;
    x1: number;
    y1: number;
    size: number;
  };
}

export interface CroppedSquare {
  data: Uint8ClampedArray;
  size: number;
  box: LineBox;
  squareBox: { x0: number; y0: number; x1: number; y1: number; size: number };
  scale: number;
}

/**
 * Crops a single line from the flattened grayscale buffer, preserving valley-aware
 * grayscale context around ascenders/descenders and filling outer padding with white (255).
 */
export function cropLine(
  gray: Uint8ClampedArray,
  w: number,
  h: number,
  box: LineBox,
  pad = 16,
  options?: { contextPadX?: number; contextPadY?: number }
): CroppedLine {
  const cropW = Math.max(1, box.x1 - box.x0 + 1 + pad * 2);
  const cropH = Math.max(1, box.y1 - box.y0 + 1 + pad * 2);
  const data = new Uint8ClampedArray(cropW * cropH);
  data.fill(255);

  const baseX0 = Math.max(0, box.x0);
  const baseX1 = Math.min(w - 1, box.x1);
  const baseY0 = Math.max(0, box.y0);
  const baseY1 = Math.min(h - 1, box.y1);
  if (w <= 0 || h <= 0 || baseX1 < baseX0 || baseY1 < baseY0) {
    return { data, width: cropW, height: cropH, box };
  }

  const boxW = baseX1 - baseX0 + 1;
  const boxH = baseY1 - baseY0 + 1;
  const maxCtxY = Math.max(0, options?.contextPadY ?? pad);
  const maxCtxX = Math.max(0, options?.contextPadX ?? Math.min(pad, 6));

  const countRowDark = (y: number): number => {
    const rowOff = y * w;
    let darkCount = 0;
    for (let x = baseX0; x <= baseX1; x++) {
      if (gray[rowOff + x] < 180) darkCount++;
    }
    return darkCount;
  };

  const countColDark = (x: number): number => {
    let darkCount = 0;
    for (let y = baseY0; y <= baseY1; y++) {
      if (gray[y * w + x] < 180) darkCount++;
    }
    return darkCount;
  };

  const denseRowThresh = Math.max(6, Math.min(10, Math.round(boxW * 0.15)));
  const sparseRowThresh = Math.max(4, Math.floor(denseRowThresh * 0.45));
  const edgeProbeRows = Math.min(3, boxH);

  let ctxY0 = baseY0;
  const minBoundY = Math.max(0, box.y0 - maxCtxY);
  if (options?.contextPadY !== undefined) {
    ctxY0 = Math.min(baseY0, minBoundY);
  } else {
    let whiteGapRows = 0;
    let seenCleanGap = false;
    let gapStopY = baseY0;
    let inDarkRunAfterGap = false;

    let sparseValleyRows = 0;
    let seenSparseValley = false;
    let valleyMinDark = Infinity;
    let valleyStopY = baseY0;
    let denseAfterSparse = 0;
    let lastDarkCount = 0;

    // Seed valley state if the outer margin rows inside box already lie in an inter-line valley
    for (let ey = 0; ey < edgeProbeRows; ey++) {
      const dc = countRowDark(baseY0 + ey);
      if (dc === 0) {
        whiteGapRows++;
      } else {
        break;
      }
    }
    if (whiteGapRows >= 2) {
      seenCleanGap = true;
      gapStopY = baseY0;
    }
    for (let ey = 0; ey < edgeProbeRows; ey++) {
      const dc = countRowDark(baseY0 + ey);
      if (dc <= sparseRowThresh) {
        sparseValleyRows++;
        if (dc <= valleyMinDark) {
          valleyMinDark = dc;
          valleyStopY = baseY0 + ey;
        }
      } else {
        break;
      }
    }
    if (sparseValleyRows >= 2) {
      seenSparseValley = true;
    }

    for (let y = baseY0 - 1; y >= minBoundY; y--) {
      const darkCount = countRowDark(y);
      lastDarkCount = darkCount;

      if (darkCount === 0) {
        whiteGapRows++;
        sparseValleyRows++;
        inDarkRunAfterGap = false;
        denseAfterSparse = 0;
        if (whiteGapRows >= 2) {
          seenCleanGap = true;
          gapStopY = y;
        }
        valleyMinDark = 0;
        valleyStopY = y;
        ctxY0 = y;
      } else if (seenCleanGap) {
        inDarkRunAfterGap = true;
        if (darkCount >= denseRowThresh) {
          ctxY0 = gapStopY;
          break;
        }
        ctxY0 = y;
      } else {
        whiteGapRows = 0;
        if (darkCount <= sparseRowThresh) {
          sparseValleyRows++;
          if (sparseValleyRows >= 2) seenSparseValley = true;
          if (darkCount <= valleyMinDark) {
            valleyMinDark = darkCount;
            valleyStopY = y;
          }
          denseAfterSparse = 0;
          ctxY0 = y;
        } else if (seenSparseValley) {
          if (darkCount >= denseRowThresh) {
            denseAfterSparse++;
            if (denseAfterSparse >= 2 || darkCount >= Math.round(denseRowThresh * 1.3)) {
              ctxY0 = valleyStopY;
              break;
            }
          } else {
            denseAfterSparse = 0;
          }
          ctxY0 = y;
        } else {
          ctxY0 = y;
        }
      }
    }

    if (seenCleanGap && inDarkRunAfterGap) {
      ctxY0 = gapStopY;
    } else if (
      seenSparseValley &&
      !seenCleanGap &&
      (denseAfterSparse >= 1 || lastDarkCount >= Math.max(Math.round(denseRowThresh * 0.7), sparseRowThresh + 3))
    ) {
      ctxY0 = valleyStopY;
    }
    ctxY0 = Math.min(baseY0, ctxY0);
  }

  let ctxY1 = baseY1;
  const maxBoundY = Math.min(h - 1, box.y1 + maxCtxY);
  if (options?.contextPadY !== undefined) {
    ctxY1 = Math.max(baseY1, maxBoundY);
  } else {
    let whiteGapRows = 0;
    let seenCleanGap = false;
    let gapStopY = baseY1;
    let inDarkRunAfterGap = false;

    let sparseValleyRows = 0;
    let seenSparseValley = false;
    let valleyMinDark = Infinity;
    let valleyStopY = baseY1;
    let denseAfterSparse = 0;
    let lastDarkCount = 0;

    for (let ey = 0; ey < edgeProbeRows; ey++) {
      const dc = countRowDark(baseY1 - ey);
      if (dc === 0) {
        whiteGapRows++;
      } else {
        break;
      }
    }
    if (whiteGapRows >= 2) {
      seenCleanGap = true;
      gapStopY = baseY1;
    }
    for (let ey = 0; ey < edgeProbeRows; ey++) {
      const dc = countRowDark(baseY1 - ey);
      if (dc <= sparseRowThresh) {
        sparseValleyRows++;
        if (dc <= valleyMinDark) {
          valleyMinDark = dc;
          valleyStopY = baseY1 - ey;
        }
      } else {
        break;
      }
    }
    if (sparseValleyRows >= 2) {
      seenSparseValley = true;
    }

    for (let y = baseY1 + 1; y <= maxBoundY; y++) {
      const darkCount = countRowDark(y);
      lastDarkCount = darkCount;

      if (darkCount === 0) {
        whiteGapRows++;
        sparseValleyRows++;
        inDarkRunAfterGap = false;
        denseAfterSparse = 0;
        if (whiteGapRows >= 2) {
          seenCleanGap = true;
          gapStopY = y;
        }
        valleyMinDark = 0;
        valleyStopY = y;
        ctxY1 = y;
      } else if (seenCleanGap) {
        inDarkRunAfterGap = true;
        if (darkCount >= denseRowThresh) {
          ctxY1 = gapStopY;
          break;
        }
        ctxY1 = y;
      } else {
        whiteGapRows = 0;
        if (darkCount <= sparseRowThresh) {
          sparseValleyRows++;
          if (sparseValleyRows >= 2) seenSparseValley = true;
          if (darkCount <= valleyMinDark) {
            valleyMinDark = darkCount;
            valleyStopY = y;
          }
          denseAfterSparse = 0;
          ctxY1 = y;
        } else if (seenSparseValley) {
          if (darkCount >= denseRowThresh) {
            denseAfterSparse++;
            if (denseAfterSparse >= 2 || darkCount >= Math.round(denseRowThresh * 1.3)) {
              ctxY1 = valleyStopY;
              break;
            }
          } else {
            denseAfterSparse = 0;
          }
          ctxY1 = y;
        } else {
          ctxY1 = y;
        }
      }
    }

    if (seenCleanGap && inDarkRunAfterGap) {
      ctxY1 = gapStopY;
    } else if (
      seenSparseValley &&
      !seenCleanGap &&
      (denseAfterSparse >= 1 || lastDarkCount >= Math.max(Math.round(denseRowThresh * 0.7), sparseRowThresh + 3))
    ) {
      ctxY1 = valleyStopY;
    }
    ctxY1 = Math.max(baseY1, ctxY1);
  }

  let ctxX0 = baseX0;
  const minBoundX = Math.max(0, box.x0 - maxCtxX);
  if (options?.contextPadX !== undefined) {
    ctxX0 = Math.min(baseX0, minBoundX);
  } else {
    const denseColThresh = Math.max(3, Math.round(boxH * 0.15));
    let whiteGapCols = 0;
    let seenCleanGap = false;
    let gapStopX = baseX0;
    let inDarkRunAfterGap = false;
    for (let x = baseX0 - 1; x >= minBoundX; x--) {
      const darkCount = countColDark(x);
      if (darkCount === 0) {
        whiteGapCols++;
        inDarkRunAfterGap = false;
        if (whiteGapCols >= 2) {
          seenCleanGap = true;
          gapStopX = x;
        }
        ctxX0 = x;
      } else if (seenCleanGap) {
        inDarkRunAfterGap = true;
        if (darkCount >= denseColThresh) {
          ctxX0 = gapStopX;
          break;
        }
        ctxX0 = x;
      } else {
        whiteGapCols = 0;
        ctxX0 = x;
      }
    }
    if (seenCleanGap && inDarkRunAfterGap) {
      ctxX0 = gapStopX;
    }
  }

  let ctxX1 = baseX1;
  const maxBoundX = Math.min(w - 1, box.x1 + maxCtxX);
  if (options?.contextPadX !== undefined) {
    ctxX1 = Math.max(baseX1, maxBoundX);
  } else {
    const denseColThresh = Math.max(3, Math.round(boxH * 0.15));
    let whiteGapCols = 0;
    let seenCleanGap = false;
    let gapStopX = baseX1;
    let inDarkRunAfterGap = false;
    for (let x = baseX1 + 1; x <= maxBoundX; x++) {
      const darkCount = countColDark(x);
      if (darkCount === 0) {
        whiteGapCols++;
        inDarkRunAfterGap = false;
        if (whiteGapCols >= 2) {
          seenCleanGap = true;
          gapStopX = x;
        }
        ctxX1 = x;
      } else if (seenCleanGap) {
        inDarkRunAfterGap = true;
        if (darkCount >= denseColThresh) {
          ctxX1 = gapStopX;
          break;
        }
        ctxX1 = x;
      } else {
        whiteGapCols = 0;
        ctxX1 = x;
      }
    }
    if (seenCleanGap && inDarkRunAfterGap) {
      ctxX1 = gapStopX;
    }
  }

  for (let y = ctxY0; y <= ctxY1; y++) {
    const dstY = y - box.y0 + pad;
    if (dstY < 0 || dstY >= cropH) continue;
    const srcRow = y * w;
    const dstRow = dstY * cropW;
    for (let x = ctxX0; x <= ctxX1; x++) {
      const dstX = x - box.x0 + pad;
      if (dstX < 0 || dstX >= cropW) continue;
      data[dstRow + dstX] = gray[srcRow + x];
    }
  }

  return { data, width: cropW, height: cropH, box };
}

/**
 * Segments a binarized document or line region into individual word bounding boxes,
 * detecting the exact tight ink edges of each word, and computing a square bounding box
 * that fits the word centered with equal width & height.
 */
export function segmentWords(
  binary: Uint8ClampedArray,
  w: number,
  h: number,
  explicitLines?: LineBox[]
): WordBox[] {
  if (w === 0 || h === 0) return [];

  const lines =
    explicitLines && explicitLines.length > 0 ? explicitLines : segmentLines(binary, w, h);
  if (lines.length === 0) return [];

  const activeBinary =
    w >= 280 && h >= 200 ? filterTextComponents(binary, w, h) : binary;
  const words: WordBox[] = [];

  lines.forEach((line, lineIdx) => {
    const lineW = line.x1 - line.x0 + 1;
    const lineH = Math.max(1, line.y1 - line.y0 + 1);
    if (lineW <= 4 || lineH <= 4) return;

    // Projection of ink columns within this line
    const colInk = new Int32Array(lineW);
    let totalLineInk = 0;

    for (let y = Math.max(0, line.y0); y <= Math.min(h - 1, line.y1); y++) {
      const rowOffset = y * w;
      for (let x = line.x0; x <= line.x1; x++) {
        if (activeBinary[rowOffset + x] === 0) {
          colInk[x - line.x0]++;
          totalLineInk++;
        }
      }
    }

    if (totalLineInk < 6) return;

    // Collect non-zero gaps to compute adaptive inter-word spacing threshold
    const gaps: number[] = [];
    let inInk = false;
    let gapStart = -1;

    for (let i = 0; i < lineW; i++) {
      if (colInk[i] > 0) {
        if (!inInk) {
          inInk = true;
          if (gapStart !== -1) {
            const gapLen = i - gapStart;
            if (gapLen > 0) gaps.push(gapLen);
          }
        }
      } else {
        if (inInk) {
          inInk = false;
          gapStart = i;
        }
      }
    }

    // Typical word gap is ~0.22*lineH or greater than inter-letter kerning
    let wordGapThreshold = Math.max(5, Math.round(lineH * 0.22));
    if (gaps.length >= 4) {
      gaps.sort((a, b) => a - b);
      const medianGap = gaps[Math.floor(gaps.length * 0.5)];
      const p80Gap = gaps[Math.floor(gaps.length * 0.8)];
      if (p80Gap >= medianGap * 1.8 && p80Gap >= 4) {
        wordGapThreshold = Math.max(wordGapThreshold, Math.round(medianGap * 1.5));
      }
    }

    // Cluster columns into words based on wordGapThreshold
    interface WordColSpan {
      startX: number;
      endX: number;
    }
    const spans: WordColSpan[] = [];
    let curSpanStart = -1;
    let curSpanEnd = -1;
    let currentZeroCount = 0;

    for (let i = 0; i < lineW; i++) {
      if (colInk[i] > 0) {
        if (curSpanStart === -1) {
          curSpanStart = i;
        } else if (currentZeroCount >= wordGapThreshold) {
          spans.push({ startX: curSpanStart, endX: curSpanEnd });
          curSpanStart = i;
        }
        curSpanEnd = i;
        currentZeroCount = 0;
      } else if (curSpanStart !== -1) {
        currentZeroCount++;
      }
    }
    if (curSpanStart !== -1 && curSpanEnd >= curSpanStart) {
      spans.push({ startX: curSpanStart, endX: curSpanEnd });
    }

    let wordInLineIdx = 0;
    for (const span of spans) {
      const absStartX = line.x0 + span.startX;
      const absEndX = line.x0 + span.endX;

      // Tighten ink bounding box vertically and horizontally
      let minX = absEndX;
      let maxX = absStartX;
      let minY = line.y1;
      let maxY = line.y0;
      let inkCount = 0;

      for (let y = Math.max(0, line.y0); y <= Math.min(h - 1, line.y1); y++) {
        const rowOff = y * w;
        for (let x = absStartX; x <= absEndX; x++) {
          if (activeBinary[rowOff + x] === 0) {
            inkCount++;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
        }
      }

      if (inkCount < 5 || maxX < minX || maxY < minY) continue;

      const wordW = maxX - minX + 1;
      const wordH = maxY - minY + 1;
      if (wordW < 3 && wordH < 3) continue;

      // Fit to square with proportional padding
      const pad = Math.round(Math.max(6, Math.min(wordW, wordH) * 0.18));
      const side = Math.max(wordW + pad * 2, wordH + pad * 2);
      const cx = (minX + maxX) / 2;
      const cy = (minY + maxY) / 2;
      const sqX0 = Math.round(cx - side / 2);
      const sqY0 = Math.round(cy - side / 2);

      words.push({
        x0: minX,
        y0: minY,
        x1: maxX,
        y1: maxY,
        lineIndex: lineIdx,
        wordIndex: wordInLineIdx++,
        squareBox: {
          x0: sqX0,
          y0: sqY0,
          x1: sqX0 + side - 1,
          y1: sqY0 + side - 1,
          size: side,
        },
      });
    }
  });

  return words;
}

/**
 * Crops any bounding box (word, symbol, or text line) from a grayscale buffer and fits
 * it cleanly into a square without distortion, centered with white paper background (255),
 * optionally resampling to a fixed targetSize (e.g. 384x384 for vision encoders).
 */
export function cropToSquare(
  gray: Uint8ClampedArray,
  w: number,
  h: number,
  box: LineBox,
  targetSize?: number,
  padRatio = 0.18
): CroppedSquare {
  const boxW = Math.max(1, box.x1 - box.x0 + 1);
  const boxH = Math.max(1, box.y1 - box.y0 + 1);
  const pad = Math.round(Math.max(6, Math.max(boxW, boxH) * padRatio));
  const side = Math.max(boxW, boxH) + pad * 2;

  const cx = (box.x0 + box.x1) / 2;
  const cy = (box.y0 + box.y1) / 2;
  const sqX0 = Math.round(cx - side / 2);
  const sqY0 = Math.round(cy - side / 2);

  const outSize = targetSize ? Math.max(24, Math.round(targetSize)) : side;
  const out = new Uint8ClampedArray(outSize * outSize);
  out.fill(255);

  const scale = outSize / side;

  for (let dy = 0; dy < outSize; dy++) {
    const sy = sqY0 + dy / scale;
    const y0 = Math.floor(sy);
    const fy = sy - y0;
    const dstRow = dy * outSize;

    for (let dx = 0; dx < outSize; dx++) {
      const sx = sqX0 + dx / scale;
      const x0 = Math.floor(sx);
      const fx = sx - x0;

      if (x0 < 0 || x0 >= w || y0 < 0 || y0 >= h) {
        out[dstRow + dx] = 255;
        continue;
      }

      const x1 = Math.min(w - 1, x0 + 1);
      const y1 = Math.min(h - 1, y0 + 1);

      const i00 = gray[y0 * w + x0];
      const i10 = gray[y0 * w + x1];
      const i01 = gray[y1 * w + x0];
      const i11 = gray[y1 * w + x1];

      const top = i00 + (i10 - i00) * fx;
      const bottom = i01 + (i11 - i01) * fx;
      out[dstRow + dx] = Math.round(top + (bottom - top) * fy);
    }
  }

  return {
    data: out,
    size: outSize,
    box,
    squareBox: {
      x0: sqX0,
      y0: sqY0,
      x1: sqX0 + side - 1,
      y1: sqY0 + side - 1,
      size: side,
    },
    scale,
  };
}

export interface HandwrittenWordCrop {
  data: Uint8ClampedArray;
  width: number;
  height: number;
  box: LineBox;
}

/**
 * Removes disconnected ascender/descender bleeds from adjacent lines touching the
 * top/bottom borders of a LineBox while keeping small dots and core line strokes intact.
 */
export function stripLineEdgeBleeds(
  binary: Uint8ClampedArray,
  w: number,
  h: number,
  box: LineBox
): { mask: Uint8ClampedArray; width: number; height: number } {
  const x0 = Math.max(0, box.x0);
  const y0 = Math.max(0, box.y0);
  const x1 = Math.min(w - 1, box.x1);
  const y1 = Math.min(h - 1, box.y1);
  const lw = Math.max(1, x1 - x0 + 1);
  const lh = Math.max(1, y1 - y0 + 1);

  const mask = new Uint8ClampedArray(lw * lh).fill(255);
  const vis = new Uint8Array(lw * lh);
  const q: number[] = [];
  const midTop = Math.round(lh * 0.28);
  const midBot = Math.round(lh * 0.78);

  const ccs: Array<{
    minLX: number;
    maxLX: number;
    minLY: number;
    maxLY: number;
    pts: number[];
  }> = [];

  for (let ly = 0; ly < lh; ly++) {
    for (let lx = 0; lx < lw; lx++) {
      const liIdx = ly * lw + lx;
      if (vis[liIdx] || binary[(y0 + ly) * w + (x0 + lx)] !== 0) continue;

      vis[liIdx] = 1;
      q.length = 0;
      q.push(liIdx);
      let head = 0;
      let minLY = lh;
      let maxLY = -1;
      let minLX = lw;
      let maxLX = -1;

      while (head < q.length) {
        const idx = q[head++];
        const cy = Math.floor(idx / lw);
        const cx = idx - cy * lw;
        if (cy < minLY) minLY = cy;
        if (cy > maxLY) maxLY = cy;
        if (cx < minLX) minLX = cx;
        if (cx > maxLX) maxLX = cx;

        for (let dy = -1; dy <= 1; dy++) {
          const ny = cy + dy;
          if (ny < 0 || ny >= lh) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const nx = cx + dx;
            if (nx < 0 || nx >= lw) continue;
            const ni = ny * lw + nx;
            if (!vis[ni] && binary[(y0 + ny) * w + (x0 + nx)] === 0) {
              vis[ni] = 1;
              q.push(ni);
            }
          }
        }
      }

      ccs.push({ minLX, maxLX, minLY, maxLY, pts: q.slice() });
    }
  }

  const isoGap = Math.max(10, Math.round(lh * 0.24));
  for (let i = 0; i < ccs.length; i++) {
    const cc = ccs[i];
    const cw = cc.maxLX - cc.minLX + 1;
    const ch = cc.maxLY - cc.minLY + 1;
    const touchesMid = cc.maxLY >= midTop && cc.minLY <= midBot;
    const isUpperBleedLoop = cc.minLY <= 2 && cc.maxLY < lh * 0.42 && cw > 10;
    const isLowerBleedLoop = cc.maxLY >= lh - 3 && cc.minLY > lh * 0.62 && cw > 10;

    // Only drop a thin vertical stem if it bleeds in from top/bottom line edge AND sits
    // horizontally isolated from neighboring glyphs (never drop valid first/last letters).
    let isIsolatedBorderStem = false;
    if (
      (cc.minLY <= 1 || cc.maxLY >= lh - 2) &&
      cw <= 6 &&
      ch >= lh * 0.45 &&
      cw / Math.max(1, ch) < 0.28
    ) {
      const hasAdjacentInk = ccs.some(
        (other, j) =>
          j !== i && other.maxLX >= cc.minLX - isoGap && other.minLX <= cc.maxLX + isoGap
      );
      if (!hasAdjacentInk) {
        isIsolatedBorderStem = true;
      }
    }

    if (
      !isIsolatedBorderStem &&
      !isUpperBleedLoop &&
      !isLowerBleedLoop &&
      (touchesMid || cc.pts.length <= 28)
    ) {
      for (let k = 0; k < cc.pts.length; k++) {
        mask[cc.pts[k]] = 0;
      }
    }
  }

  return { mask, width: lw, height: lh };
}

/**
 * Segments a handwritten line into individual word crops with adjacent-line bleeds
 * stripped and faint grayscale ink strokes bridged across binarization dropouts.
 */
export function segmentHandwrittenWordsInLine(
  binary: Uint8ClampedArray,
  gray: Uint8ClampedArray,
  w: number,
  h: number,
  box: LineBox,
  pad = 2
): HandwrittenWordCrop[] {
  const x0 = Math.max(0, box.x0);
  const y0 = Math.max(0, box.y0);
  const { mask, width: lw, height: lh } = stripLineEdgeBleeds(binary, w, h, box);

  const colInk = new Int32Array(lw);
  let totalLineInk = 0;
  for (let ly = 0; ly < lh; ly++) {
    const mOff = ly * lw;
    for (let lx = 0; lx < lw; lx++) {
      if (mask[mOff + lx] === 0) {
        colInk[lx]++;
        totalLineInk++;
      }
    }
  }

  if (totalLineInk < 14) return [];

  // Check if a horizontal column range actually contains faint gray pen ink (< 226)
  // in the core band so faint pen strokes don't fracture single words into fragments.
  const hasFaintGrayInGap = (gxStart: number, gxEnd: number): boolean => {
    const gapW = gxEnd - gxStart + 1;
    if (gapW <= 0 || gapW > Math.max(18, Math.round(lh * 0.45))) return false;
    let faintCols = 0;
    const bandTop = Math.round(lh * 0.18);
    const bandBot = Math.round(lh * 0.84);
    for (let lx = gxStart; lx <= gxEnd; lx++) {
      const gx = x0 + lx;
      let darkPx = 0;
      for (let ly = bandTop; ly <= bandBot; ly++) {
        if (gray[(y0 + ly) * w + gx] < 226) darkPx++;
      }
      if (darkPx >= 1) faintCols++;
    }
    return faintCols / gapW >= 0.28;
  };

  // Collect gap widths to adaptively distinguish inter-letter kerning from word spacing
  const gaps: number[] = [];
  let inGapInk = false;
  let gapStartX = -1;
  for (let i = 0; i < lw; i++) {
    if (colInk[i] > 0) {
      if (!inGapInk) {
        inGapInk = true;
        if (gapStartX !== -1) {
          const gl = i - gapStartX;
          if (gl > 0) gaps.push(gl);
        }
      }
    } else if (inGapInk) {
      inGapInk = false;
      gapStartX = i;
    }
  }

  let wordGap = Math.max(9, Math.round(lh * 0.27));
  if (gaps.length >= 4) {
    const sortedGaps = [...gaps].sort((a, b) => a - b);
    const medianGap = sortedGaps[Math.floor(sortedGaps.length * 0.5)];
    const p80Gap = sortedGaps[Math.floor(sortedGaps.length * 0.8)];
    if (p80Gap >= medianGap * 2.2 && p80Gap >= 6) {
      wordGap = Math.max(
        7,
        Math.min(Math.round(lh * 0.34), Math.max(Math.round(medianGap * 2.2), Math.round(lh * 0.21)))
      );
    }
  }
  const rawSpans: Array<{ sx: number; ex: number; ink: number }> = [];
  let s = -1;
  let e = -1;
  let z = 0;
  let spanInk = 0;

  for (let i = 0; i < lw; i++) {
    if (colInk[i] > 0) {
      if (s === -1) {
        s = i;
        spanInk = colInk[i];
      } else if (z >= wordGap && !hasFaintGrayInGap(e + 1, i - 1)) {
        rawSpans.push({ sx: s, ex: e, ink: spanInk });
        s = i;
        spanInk = colInk[i];
      } else {
        spanInk += colInk[i];
      }
      e = i;
      z = 0;
    } else if (s !== -1) {
      z++;
    }
  }
  if (s !== -1) {
    rawSpans.push({ sx: s, ex: e, ink: spanInk });
  }

  const keptSpans = rawSpans.filter(
    (sp) => sp.ink >= Math.max(14, Math.round(totalLineInk * 0.04))
  );

  const crops: HandwrittenWordCrop[] = [];
  for (const sp of keptSpans) {
    let minX = x0 + sp.ex;
    let maxX = x0 + sp.sx;
    let minY = y0 + lh - 1;
    let maxY = y0;

    for (let ly = 0; ly < lh; ly++) {
      const mOff = ly * lw;
      for (let lx = sp.sx; lx <= sp.ex; lx++) {
        if (mask[mOff + lx] === 0) {
          const gx = x0 + lx;
          const gy = y0 + ly;
          if (gx < minX) minX = gx;
          if (gx > maxX) maxX = gx;
          if (gy < minY) minY = gy;
          if (gy > maxY) maxY = gy;
        }
      }
    }

    if (maxX < minX || maxY < minY) continue;
    const cw = maxX - minX + 1;
    const ch = maxY - minY + 1;
    const outW = cw + pad * 2;
    const outH = ch + pad * 2;
    const wordGray = new Uint8ClampedArray(outW * outH).fill(255);

    for (let gy = minY; gy <= maxY; gy++) {
      const srcRow = gy * w;
      const dstRow = (gy - minY + pad) * outW;
      for (let gx = minX; gx <= maxX; gx++) {
        wordGray[dstRow + (gx - minX + pad)] = gray[srcRow + gx];
      }
    }

    crops.push({
      data: wordGray,
      width: outW,
      height: outH,
      box: { x0: minX, y0: minY, x1: maxX, y1: maxY },
    });
  }

  return crops;
}
