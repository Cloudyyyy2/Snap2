import { filterTextComponents } from './cleanup';

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
  h: number
): LineBox[] {
  if (w === 0 || h === 0) return [];

  const activeBinary =
    w >= 280 && h >= 200 ? filterTextComponents(binary, w, h) : binary;

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
    const hasDeepValley = minBetween <= weakerPeak * 0.35;

    if (p - prev < minPeakDistance && !hasDeepValley) {
      if (smooth[p] > smooth[prev]) {
        peaks[peaks.length - 1] = p;
      }
    } else if (!hasDeepValley) {
      if (smooth[p] > smooth[prev]) {
        peaks[peaks.length - 1] = p;
      }
    } else {
      peaks.push(p);
    }
  }

  if (peaks.length === 0) return [];

  // Determine vertical boundaries between peaks by finding the lowest valley between adjacent peaks
  const lineRanges: Array<{ y0: number; y1: number }> = [];
  const activeThreshold = Math.max(1, maxProj * 0.04);

  for (let i = 0; i < peaks.length; i++) {
    let y0 = 0;
    if (i === 0) {
      y0 = peaks[0];
      while (y0 > 0 && smooth[y0 - 1] >= activeThreshold) y0--;
    } else {
      const prevPeak = peaks[i - 1];
      let minY = prevPeak;
      let minVal = Infinity;
      for (let y = prevPeak + 1; y < peaks[i]; y++) {
        if (smooth[y] < minVal) {
          minVal = smooth[y];
          minY = y;
        }
      }
      y0 = minY;
    }

    let y1 = h - 1;
    if (i === peaks.length - 1) {
      y1 = peaks[i];
      while (y1 < h - 1 && smooth[y1 + 1] >= activeThreshold) y1++;
    } else {
      const nextPeak = peaks[i + 1];
      let minY = peaks[i];
      let minVal = Infinity;
      for (let y = peaks[i] + 1; y < nextPeak; y++) {
        if (smooth[y] < minVal) {
          minVal = smooth[y];
          minY = y;
        }
      }
      y1 = minY;
    }

    lineRanges.push({ y0, y1 });
  }

  // Refine x0..x1 using vertical column density to drop distant edge smudges and tiny noise bands
  const boxes: LineBox[] = [];
  const colInk = new Int32Array(w);
  const largeGap = Math.max(45, Math.round(w * 0.14));

  for (const range of lineRanges) {
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
    const segments: Array<{ x0: number; x1: number; ink: number }> = [];
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

    // Keep meaningful segments (drop isolated tiny edge specks far from main text)
    const significant = segments.filter(
      (s) => s.ink >= Math.max(12, totalInk * 0.08) || segments.length === 1
    );
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

    const boxH = maxY - minY + 1;
    const boxW = maxX - minX + 1;
    const minBoxW = w >= 280 ? 24 : 8;
    const minBoxH = h >= 200 ? 12 : 6;
    const minInk = w >= 280 ? 40 : 12;
    if (maxX >= minX && boxH >= minBoxH && boxW >= minBoxW && inkPixels >= minInk) {
      boxes.push({ x0: minX, y0: minY, x1: maxX, y1: maxY });
    }
  }

  return boxes;
}

/**
 * Splits overly wide line boxes (aspect > maxAspect) at deep inter-word whitespace gaps
 * so downstream fixed-square vision encoders (like TrOCR's 384x384 input) do not squash characters.
 */
export function splitWideLineBox(
  binary: Uint8ClampedArray,
  w: number,
  h: number,
  box: LineBox,
  maxAspect = 5.0
): LineBox[] {
  const boxW = box.x1 - box.x0 + 1;
  const boxH = Math.max(1, box.y1 - box.y0 + 1);
  if (boxW / boxH <= maxAspect || boxW < 180) {
    return [box];
  }

  const maxChunkW = Math.max(120, Math.round(boxH * maxAspect));
  const minSplitGap = Math.max(6, Math.round(boxH * 0.22));
  const colInk = new Int32Array(boxW);

  for (let y = Math.max(0, box.y0); y <= Math.min(h - 1, box.y1); y++) {
    const rowOffset = y * w;
    for (let x = box.x0; x <= box.x1; x++) {
      if (binary[rowOffset + x] === 0) {
        colInk[x - box.x0]++;
      }
    }
  }

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

    // Look backward from cursor + maxChunkW for the widest whitespace gap
    const searchStart = Math.min(boxW - 1, cursor + Math.round(maxChunkW * 0.55));
    const searchEnd = Math.min(boxW - 1, cursor + maxChunkW);
    let bestGapCenter = -1;
    let bestGapWidth = 0;

    let gapStart = -1;
    for (let lx = searchStart; lx <= searchEnd; lx++) {
      if (colInk[lx] <= 1) {
        if (gapStart === -1) gapStart = lx;
      } else if (gapStart !== -1) {
        const gw = lx - gapStart;
        if (gw >= minSplitGap && gw >= bestGapWidth) {
          bestGapWidth = gw;
          bestGapCenter = Math.floor((gapStart + lx - 1) / 2);
        }
        gapStart = -1;
      }
    }

    const splitAt = bestGapCenter !== -1 ? bestGapCenter : searchEnd;
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

/**
 * Crops a single line from the flattened grayscale buffer with clean white padding.
 */
export function cropLine(
  gray: Uint8ClampedArray,
  w: number,
  h: number,
  box: LineBox,
  pad = 16
): CroppedLine {
  const cropW = Math.max(1, box.x1 - box.x0 + 1 + pad * 2);
  const cropH = Math.max(1, box.y1 - box.y0 + 1 + pad * 2);
  const data = new Uint8ClampedArray(cropW * cropH);
  data.fill(255);

  for (let y = box.y0; y <= box.y1; y++) {
    if (y < 0 || y >= h) continue;
    const dstY = y - box.y0 + pad;
    const srcRow = y * w;
    const dstRow = dstY * cropW;
    for (let x = box.x0; x <= box.x1; x++) {
      if (x < 0 || x >= w) continue;
      data[dstRow + (x - box.x0 + pad)] = gray[srcRow + x];
    }
  }

  return { data, width: cropW, height: cropH, box };
}
