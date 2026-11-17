/**
 * Pure TypeScript grayscale image cleanup for everyday handwriting on paper.
 * Operates on Uint8ClampedArray (1 byte per pixel, 0 = black ink, 255 = white paper),
 * usable identically in both Node.js and browser environments.
 */

function buildIntegralImage(data: Uint8ClampedArray, w: number, h: number): Float64Array {
  const stride = w + 1;
  const integral = new Float64Array(stride * (h + 1));

  for (let y = 0; y < h; y++) {
    let rowSum = 0;
    const rowOffset = y * w;
    const prevIntRow = y * stride;
    const curIntRow = (y + 1) * stride;
    for (let x = 0; x < w; x++) {
      rowSum += data[rowOffset + x];
      integral[curIntRow + x + 1] = integral[prevIntRow + x + 1] + rowSum;
    }
  }

  return integral;
}

function buildSquaredIntegralImage(data: Uint8ClampedArray, w: number, h: number): Float64Array {
  const stride = w + 1;
  const integral = new Float64Array(stride * (h + 1));

  for (let y = 0; y < h; y++) {
    let rowSum = 0;
    const rowOffset = y * w;
    const prevIntRow = y * stride;
    const curIntRow = (y + 1) * stride;
    for (let x = 0; x < w; x++) {
      const v = data[rowOffset + x];
      rowSum += v * v;
      integral[curIntRow + x + 1] = integral[prevIntRow + x + 1] + rowSum;
    }
  }

  return integral;
}

function boxSum(
  integral: Float64Array,
  w: number,
  h: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number
): { sum: number; count: number } {
  const cx0 = Math.max(0, x0);
  const cy0 = Math.max(0, y0);
  const cx1 = Math.min(w - 1, x1);
  const cy1 = Math.min(h - 1, y1);
  const stride = w + 1;

  const sum =
    integral[(cy1 + 1) * stride + (cx1 + 1)] -
    integral[cy0 * stride + (cx1 + 1)] -
    integral[(cy1 + 1) * stride + cx0] +
    integral[cy0 * stride + cx0];

  const count = Math.max(1, (cx1 - cx0 + 1) * (cy1 - cy0 + 1));
  return { sum, count };
}

/**
 * Estimates local paper brightness with a large box blur (radius ~ 1/20 of min dimension)
 * and normalizes: out = clamp(255 * gray / background).
 * Removes camera shadows and uneven lighting while keeping pen strokes crisp on white paper.
 */
export function flattenBackground(
  gray: Uint8ClampedArray,
  w: number,
  h: number
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h);
  if (w === 0 || h === 0) return out;

  const radius = Math.max(8, Math.round(Math.min(w, h) / 20));
  const integral = buildIntegralImage(gray, w, h);

  for (let y = 0; y < h; y++) {
    const y0 = y - radius;
    const y1 = y + radius;
    const rowOffset = y * w;
    for (let x = 0; x < w; x++) {
      const { sum, count } = boxSum(integral, w, h, x - radius, y0, x + radius, y1);
      const bg = Math.max(12, sum / count);
      const normalized = Math.round((255 * gray[rowOffset + x]) / bg);
      out[rowOffset + x] = normalized > 255 ? 255 : normalized < 0 ? 0 : normalized;
    }
  }

  return out;
}

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

  // Flag candidate rows that have a consistent horizontal gray band (> 60% of pixels darker than paper - 10,
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

    if (covered / w >= 0.6 && maxRun >= w * 0.35 && lineSamples.length > 0) {
      lineSamples.sort((a, b) => a - b);
      const med = lineSamples[Math.floor(lineSamples.length * 0.5)];
      // Ruled lines are typically mid/light gray/blue (not solid black ink block)
      if (med >= 60 && med <= paper - 10) {
        candidateRow[y] = 1;
        rowMedian[y] = med;
      }
    }
  }

  // Group contiguous candidate rows; ruled lines have thickness <= 3 px
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

    if (thickness <= 3) {
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
          const isCrossingInk = cur < lineTone - 28;
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
 * Sauvola local adaptive thresholding using integral images.
 * Produces binary output (0 = ink foreground, 255 = paper background)
 * intended for line segmentation and handwriting detection, NOT as TrOCR input.
 */
export function binarizeSauvola(
  gray: Uint8ClampedArray,
  w: number,
  h: number,
  windowRadius = 15,
  k = 0.2
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h);
  if (w === 0 || h === 0) return out;

  const integral = buildIntegralImage(gray, w, h);
  const sqIntegral = buildSquaredIntegralImage(gray, w, h);
  const R = 128;

  for (let y = 0; y < h; y++) {
    const y0 = y - windowRadius;
    const y1 = y + windowRadius;
    const rowOffset = y * w;
    for (let x = 0; x < w; x++) {
      const { sum, count } = boxSum(integral, w, h, x - windowRadius, y0, x + windowRadius, y1);
      const { sum: sqSum } = boxSum(sqIntegral, w, h, x - windowRadius, y0, x + windowRadius, y1);

      const mean = sum / count;
      const variance = Math.max(0, sqSum / count - mean * mean);
      const std = Math.sqrt(variance);
      const threshold = mean * (1 + k * (std / R - 1));

      out[rowOffset + x] = gray[rowOffset + x] < threshold ? 0 : 255;
    }
  }

  return out;
}
