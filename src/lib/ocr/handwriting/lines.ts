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
 * Uses smoothed horizontal ink projection and splits touching ascenders/descenders at
 * projection valleys rather than requiring strictly zero-ink gaps.
 */
export function segmentLines(
  binary: Uint8ClampedArray,
  w: number,
  h: number
): LineBox[] {
  if (w === 0 || h === 0) return [];

  const proj = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    let ink = 0;
    const rowOffset = y * w;
    for (let x = 0; x < w; x++) {
      if (binary[rowOffset + x] === 0) ink++;
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
    } else if (!hasDeepValley && minBetween > weakerPeak * 0.65) {
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

  // Refine x0..x1 and drop tiny noise bands
  const boxes: LineBox[] = [];
  for (const range of lineRanges) {
    let minX = w;
    let maxX = -1;
    let minY = range.y1;
    let maxY = range.y0;
    let inkPixels = 0;

    for (let y = range.y0; y <= range.y1; y++) {
      const rowOffset = y * w;
      for (let x = 0; x < w; x++) {
        if (binary[rowOffset + x] === 0) {
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
    if (maxX >= minX && boxH >= 6 && boxW >= 8 && inkPixels >= 12) {
      boxes.push({ x0: minX, y0: minY, x1: maxX, y1: maxY });
    }
  }

  return boxes;
}

/**
 * Crops a single line from the flattened grayscale buffer with clean white padding.
 */
export function cropLine(
  gray: Uint8ClampedArray,
  w: number,
  h: number,
  box: LineBox,
  pad = 12
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
