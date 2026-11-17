export interface TextRegionBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  inkPixels: number;
}

/**
 * Recursive XY-cut region segmentation (Stage 4: Layout & Region Segmentation).
 * Separates multi-column documents, sidebars, tables, and distinct paragraphs into
 * bounded text blocks ordered naturally top-to-bottom, left-to-right.
 */
export function segmentLayoutRegions(
  binary: Uint8ClampedArray,
  w: number,
  h: number,
  maxDepth = 4
): TextRegionBox[] {
  if (w === 0 || h === 0) return [];

  const computeInkBounds = (
    rx0: number,
    ry0: number,
    rx1: number,
    ry1: number
  ): TextRegionBox | null => {
    let minX = rx1;
    let maxX = rx0;
    let minY = ry1;
    let maxY = ry0;
    let inkPixels = 0;

    for (let y = ry0; y <= ry1; y++) {
      const rowOff = y * w;
      for (let x = rx0; x <= rx1; x++) {
        if (binary[rowOff + x] === 0) {
          inkPixels++;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    if (inkPixels < 16 || maxX < minX || maxY < minY) return null;
    return { x0: minX, y0: minY, x1: maxX, y1: maxY, inkPixels };
  };

  const findLargestZeroRun = (
    proj: Int32Array,
    minRun: number
  ): { start: number; end: number; len: number } | null => {
    let bestLen = 0;
    let bestStart = -1;
    let bestEnd = -1;
    let curStart = -1;

    for (let i = 0; i < proj.length; i++) {
      if (proj[i] === 0) {
        if (curStart === -1) curStart = i;
      } else if (curStart !== -1) {
        const len = i - curStart;
        if (len >= minRun && len > bestLen) {
          bestLen = len;
          bestStart = curStart;
          bestEnd = i - 1;
        }
        curStart = -1;
      }
    }
    return bestLen >= minRun ? { start: bestStart, end: bestEnd, len: bestLen } : null;
  };

  const results: TextRegionBox[] = [];

  const splitRegion = (box: TextRegionBox, depth: number) => {
    const rw = box.x1 - box.x0 + 1;
    const rh = box.y1 - box.y0 + 1;

    if (depth >= maxDepth || (rw < 40 && rh < 30)) {
      results.push(box);
      return;
    }

    // Check vertical column gutters first (multi-column splits require substantial horizontal gap)
    const colProj = new Int32Array(rw);
    const rowProj = new Int32Array(rh);

    for (let y = box.y0; y <= box.y1; y++) {
      const rowOff = y * w;
      for (let x = box.x0; x <= box.x1; x++) {
        if (binary[rowOff + x] === 0) {
          colProj[x - box.x0]++;
          rowProj[y - box.y0]++;
        }
      }
    }

    const minColGutter = Math.max(42, Math.round(w * 0.085));
    const minRowGutter = Math.max(30, Math.round(h * 0.08));

    const colGap = findLargestZeroRun(colProj, minColGutter);
    const rowGap = findLargestZeroRun(rowProj, minRowGutter);

    // Require meaningful ink (>= 12% of parent or >= 180px) in both halves so word gaps or stray specks don't fracture lines
    const minChildInk = Math.max(180, Math.round(box.inkPixels * 0.12));

    // Prefer column split when true side-by-side column gutter exists
    if (colGap && (!rowGap || colGap.len >= rowGap.len * 1.15)) {
      const midX = box.x0 + Math.floor((colGap.start + colGap.end) / 2);
      const left = computeInkBounds(box.x0, box.y0, midX, box.y1);
      const right = computeInkBounds(midX + 1, box.y0, box.x1, box.y1);
      if (left && right && left.inkPixels >= minChildInk && right.inkPixels >= minChildInk) {
        splitRegion(left, depth + 1);
        splitRegion(right, depth + 1);
        return;
      }
    }

    if (rowGap) {
      const midY = box.y0 + Math.floor((rowGap.start + rowGap.end) / 2);
      const top = computeInkBounds(box.x0, box.y0, box.x1, midY);
      const bottom = computeInkBounds(box.x0, midY + 1, box.x1, box.y1);
      if (top && bottom && top.inkPixels >= minChildInk && bottom.inkPixels >= minChildInk) {
        splitRegion(top, depth + 1);
        splitRegion(bottom, depth + 1);
        return;
      }
    }

    results.push(box);
  };

  const root = computeInkBounds(0, 0, w - 1, h - 1);
  if (!root) return [];

  splitRegion(root, 0);

  // Reading order sort: top-to-bottom bands (within ~24px tolerance), then left-to-right
  results.sort((a, b) => {
    if (Math.abs(a.y0 - b.y0) > 24) return a.y0 - b.y0;
    return a.x0 - b.x0;
  });

  return results;
}
