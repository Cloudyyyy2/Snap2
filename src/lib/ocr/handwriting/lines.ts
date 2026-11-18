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

