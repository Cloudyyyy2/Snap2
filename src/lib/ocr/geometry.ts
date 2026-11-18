/**
 * Pure TypeScript Geometric & Spatial Correction module for offline OCR:
 * - Projection / Hough-energy skew angle estimation and sub-degree deskew rotation
 * - Robust 4-corner document boundary fitting (line-intersection quad detection)
 * - True 8-DOF 3x3 planar projective homography solver and perspective dewarping
 */

export interface Point2D {
  x: number;
  y: number;
}

/** Ordered clockwise from top-left: [topLeft, topRight, bottomRight, bottomLeft] */
export type QuadCorners = [Point2D, Point2D, Point2D, Point2D];

export interface DeskewOptions {
  /** Maximum skew search range in degrees (+/-). Default: 18 */
  maxAngleDeg?: number;
  /** Minimum absolute angle (deg) required to apply rotation. Default: 0.4 */
  minCorrectionDeg?: number;
  /** Coarse search step in degrees. Default: 0.5 */
  coarseStepDeg?: number;
  /** Fine refinement step in degrees. Default: 0.1 */
  fineStepDeg?: number;
  /** Pixel fill intensity (0-255) for exposed corners after rotation. Default: 255 */
  backgroundFill?: number;
}

export interface DeskewResult {
  data: Uint8ClampedArray;
  angleDeg: number;
  corrected: boolean;
}

export interface PerspectiveWarpOptions {
  targetWidth?: number;
  targetHeight?: number;
  backgroundFill?: number;
}

export interface PerspectiveResult {
  data: Uint8ClampedArray;
  width: number;
  height: number;
  quad: QuadCorners | null;
  warped: boolean;
  homography: number[] | null;
}

/**
 * Orders 4 arbitrary corner points consistently as [topLeft, topRight, bottomRight, bottomLeft].
 */
export function orderQuadCorners(points: [Point2D, Point2D, Point2D, Point2D]): QuadCorners {
  const bySum = [...points].sort((a, b) => a.x + a.y - (b.x + b.y));
  const tl = bySum[0];
  const br = bySum[3];

  const remaining = [bySum[1], bySum[2]];
  // Top-right has larger (x - y), bottom-left has larger (y - x)
  remaining.sort((a, b) => a.x - a.y - (b.x - b.y));
  const bl = remaining[0];
  const tr = remaining[1];

  return [tl, tr, br, bl];
}

/**
 * Solves an 8-DOF planar projective homography H (3x3 row-major array of 9 numbers with h[8] = 1)
 * mapping 4 source points -> 4 destination points via Gaussian elimination with partial pivoting.
 */
export function solveHomography(srcQuad: QuadCorners, dstQuad: QuadCorners): number[] | null {
  // Build 8x8 linear system A * h = b where h = [h0, h1, h2, h3, h4, h5, h6, h7]^T
  const A: number[][] = Array.from({ length: 8 }, () => new Array(9).fill(0));

  for (let i = 0; i < 4; i++) {
    const sx = srcQuad[i].x;
    const sy = srcQuad[i].y;
    const dx = dstQuad[i].x;
    const dy = dstQuad[i].y;

    const r0 = i * 2;
    A[r0][0] = sx;
    A[r0][1] = sy;
    A[r0][2] = 1;
    A[r0][3] = 0;
    A[r0][4] = 0;
    A[r0][5] = 0;
    A[r0][6] = -dx * sx;
    A[r0][7] = -dx * sy;
    A[r0][8] = dx;

    const r1 = r0 + 1;
    A[r1][0] = 0;
    A[r1][1] = 0;
    A[r1][2] = 0;
    A[r1][3] = sx;
    A[r1][4] = sy;
    A[r1][5] = 1;
    A[r1][6] = -dy * sx;
    A[r1][7] = -dy * sy;
    A[r1][8] = dy;
  }

  // Gaussian elimination with partial pivoting
  for (let col = 0; col < 8; col++) {
    let pivotRow = col;
    let maxAbs = Math.abs(A[col][col]);
    for (let r = col + 1; r < 8; r++) {
      const val = Math.abs(A[r][col]);
      if (val > maxAbs) {
        maxAbs = val;
        pivotRow = r;
      }
    }

    if (maxAbs < 1e-10) {
      return null;
    }

    if (pivotRow !== col) {
      const tmp = A[col];
      A[col] = A[pivotRow];
      A[pivotRow] = tmp;
    }

    const pivot = A[col][col];
    for (let j = col; j <= 8; j++) {
      A[col][j] /= pivot;
    }

    for (let r = 0; r < 8; r++) {
      if (r === col) continue;
      const factor = A[r][col];
      if (Math.abs(factor) < 1e-12) continue;
      for (let j = col; j <= 8; j++) {
        A[r][j] -= factor * A[col][j];
      }
    }
  }

  return [A[0][8], A[1][8], A[2][8], A[3][8], A[4][8], A[5][8], A[6][8], A[7][8], 1];
}

/**
 * Applies a 3x3 row-major homography matrix H to point (x, y).
 */
export function applyHomographyPoint(H: number[], x: number, y: number): Point2D {
  const denom = H[6] * x + H[7] * y + H[8];
  if (Math.abs(denom) < 1e-9) {
    return { x, y };
  }
  return {
    x: (H[0] * x + H[1] * y + H[2]) / denom,
    y: (H[3] * x + H[4] * y + H[5]) / denom,
  };
}

/**
 * Warps a quadrilateral region in a grayscale buffer into a flat top-down rectangle
 * using an exact 8-DOF 3x3 planar projective homography and bilinear sub-pixel sampling.
 */
export function warpPerspectiveQuad(
  gray: Uint8ClampedArray,
  w: number,
  h: number,
  quad: QuadCorners,
  options: PerspectiveWarpOptions = {}
): { data: Uint8ClampedArray; width: number; height: number; homography: number[] | null } {
  const ordered = orderQuadCorners(quad);
  const [tl, tr, br, bl] = ordered;

  const topW = Math.hypot(tr.x - tl.x, tr.y - tl.y);
  const botW = Math.hypot(br.x - bl.x, br.y - bl.y);
  const leftH = Math.hypot(bl.x - tl.x, bl.y - tl.y);
  const rightH = Math.hypot(br.x - tr.x, br.y - tr.y);

  const dstW = Math.max(40, options.targetWidth ?? Math.round(Math.max(topW, botW)));
  const dstH = Math.max(40, options.targetHeight ?? Math.round(Math.max(leftH, rightH)));
  const fill = options.backgroundFill ?? 255;

  const out = new Uint8ClampedArray(dstW * dstH);
  out.fill(fill);

  if (w < 2 || h < 2) {
    return { data: out, width: dstW, height: dstH, homography: null };
  }

  const dstRect: QuadCorners = [
    { x: 0, y: 0 },
    { x: dstW - 1, y: 0 },
    { x: dstW - 1, y: dstH - 1 },
    { x: 0, y: dstH - 1 },
  ];

  // Solve inverse homography H: destination rectangle pixel (dx, dy) -> source quad pixel (sx, sy)
  const H = solveHomography(dstRect, ordered);

  if (!H) {
    // Degenerate fallback to bilinear surface interpolation
    for (let dy = 0; dy < dstH; dy++) {
      const v = dstH > 1 ? dy / (dstH - 1) : 0;
      const lx = tl.x + (bl.x - tl.x) * v;
      const ly = tl.y + (bl.y - tl.y) * v;
      const rx = tr.x + (br.x - tr.x) * v;
      const ry = tr.y + (br.y - tr.y) * v;
      const dstOff = dy * dstW;

      for (let dx = 0; dx < dstW; dx++) {
        const u = dstW > 1 ? dx / (dstW - 1) : 0;
        const sx = lx + (rx - lx) * u;
        const sy = ly + (ry - ly) * u;
        const x0 = Math.floor(sx);
        const y0 = Math.floor(sy);
        if (x0 < 0 || x0 >= w - 1 || y0 < 0 || y0 >= h - 1) continue;

        const fx = sx - x0;
        const fy = sy - y0;
        const i00 = gray[y0 * w + x0];
        const i10 = gray[y0 * w + x0 + 1];
        const i01 = gray[(y0 + 1) * w + x0];
        const i11 = gray[(y0 + 1) * w + x0 + 1];

        const top = i00 + (i10 - i00) * fx;
        const bottom = i01 + (i11 - i01) * fx;
        out[dstOff + dx] = Math.round(top + (bottom - top) * fy);
      }
    }
    return { data: out, width: dstW, height: dstH, homography: null };
  }

  const h0 = H[0];
  const h1 = H[1];
  const h2 = H[2];
  const h3 = H[3];
  const h4 = H[4];
  const h5 = H[5];
  const h6 = H[6];
  const h7 = H[7];

  for (let dy = 0; dy < dstH; dy++) {
    const rowNumX = h1 * dy + h2;
    const rowNumY = h4 * dy + h5;
    const rowDen = h7 * dy + 1;
    const dstOff = dy * dstW;

    for (let dx = 0; dx < dstW; dx++) {
      const denom = h6 * dx + rowDen;
      if (Math.abs(denom) < 1e-7) continue;

      const sx = (h0 * dx + rowNumX) / denom;
      const sy = (h3 * dx + rowNumY) / denom;

      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      if (x0 < 0 || x0 >= w - 1 || y0 < 0 || y0 >= h - 1) continue;

      const fx = sx - x0;
      const fy = sy - y0;
      const row0 = y0 * w + x0;
      const row1 = row0 + w;
      const i00 = gray[row0];
      const i10 = gray[row0 + 1];
      const i01 = gray[row1];
      const i11 = gray[row1 + 1];

      const top = i00 + (i10 - i00) * fx;
      const bottom = i01 + (i11 - i01) * fx;
      out[dstOff + dx] = Math.round(top + (bottom - top) * fy);
    }
  }

  return { data: out, width: dstW, height: dstH, homography: H };
}

/**
 * Warps a quadrilateral region in a 4-channel RGBA buffer into a flat top-down rectangle
 * using an exact 8-DOF 3x3 planar projective homography and bilinear sub-pixel sampling.
 */
export function warpPerspectiveRgba(
  rgba: Uint8ClampedArray,
  w: number,
  h: number,
  quad: QuadCorners,
  options: PerspectiveWarpOptions = {}
): { data: Uint8ClampedArray; width: number; height: number; homography: number[] | null } {
  const ordered = orderQuadCorners(quad);
  const [tl, tr, br, bl] = ordered;

  const topW = Math.hypot(tr.x - tl.x, tr.y - tl.y);
  const botW = Math.hypot(br.x - bl.x, br.y - bl.y);
  const leftH = Math.hypot(bl.x - tl.x, bl.y - tl.y);
  const rightH = Math.hypot(br.x - tr.x, br.y - tr.y);

  const dstW = Math.max(40, options.targetWidth ?? Math.round(Math.max(topW, botW)));
  const dstH = Math.max(40, options.targetHeight ?? Math.round(Math.max(leftH, rightH)));

  const out = new Uint8ClampedArray(dstW * dstH * 4);
  // Default fill: white opaque (255, 255, 255, 255)
  out.fill(255);

  if (w < 2 || h < 2) {
    return { data: out, width: dstW, height: dstH, homography: null };
  }

  const dstRect: QuadCorners = [
    { x: 0, y: 0 },
    { x: dstW - 1, y: 0 },
    { x: dstW - 1, y: dstH - 1 },
    { x: 0, y: dstH - 1 },
  ];

  const H = solveHomography(dstRect, ordered);

  if (!H) {
    // Degenerate fallback to bilinear surface interpolation
    for (let dy = 0; dy < dstH; dy++) {
      const v = dstH > 1 ? dy / (dstH - 1) : 0;
      const lx = tl.x + (bl.x - tl.x) * v;
      const ly = tl.y + (bl.y - tl.y) * v;
      const rx = tr.x + (br.x - tr.x) * v;
      const ry = tr.y + (br.y - tr.y) * v;
      const dstOff = dy * dstW * 4;

      for (let dx = 0; dx < dstW; dx++) {
        const u = dstW > 1 ? dx / (dstW - 1) : 0;
        const sx = lx + (rx - lx) * u;
        const sy = ly + (ry - ly) * u;
        const x0 = Math.floor(sx);
        const y0 = Math.floor(sy);
        if (x0 < 0 || x0 >= w - 1 || y0 < 0 || y0 >= h - 1) continue;

        const fx = sx - x0;
        const fy = sy - y0;
        const c00 = (y0 * w + x0) * 4;
        const c10 = (y0 * w + x0 + 1) * 4;
        const c01 = ((y0 + 1) * w + x0) * 4;
        const c11 = ((y0 + 1) * w + x0 + 1) * 4;
        const pixDst = dstOff + dx * 4;

        for (let c = 0; c < 4; c++) {
          const top = rgba[c00 + c] + (rgba[c10 + c] - rgba[c00 + c]) * fx;
          const bottom = rgba[c01 + c] + (rgba[c11 + c] - rgba[c01 + c]) * fx;
          out[pixDst + c] = Math.round(top + (bottom - top) * fy);
        }
      }
    }
    return { data: out, width: dstW, height: dstH, homography: null };
  }

  const h0 = H[0];
  const h1 = H[1];
  const h2 = H[2];
  const h3 = H[3];
  const h4 = H[4];
  const h5 = H[5];
  const h6 = H[6];
  const h7 = H[7];

  for (let dy = 0; dy < dstH; dy++) {
    const rowNumX = h1 * dy + h2;
    const rowNumY = h4 * dy + h5;
    const rowDen = h7 * dy + 1;
    const dstOff = dy * dstW * 4;

    for (let dx = 0; dx < dstW; dx++) {
      const denom = h6 * dx + rowDen;
      if (Math.abs(denom) < 1e-7) continue;

      const sx = (h0 * dx + rowNumX) / denom;
      const sy = (h3 * dx + rowNumY) / denom;

      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      if (x0 < 0 || x0 >= w - 1 || y0 < 0 || y0 >= h - 1) continue;

      const fx = sx - x0;
      const fy = sy - y0;
      const row0 = y0 * w;
      const row1 = (y0 + 1) * w;
      const c00 = (row0 + x0) * 4;
      const c10 = (row0 + x0 + 1) * 4;
      const c01 = (row1 + x0) * 4;
      const c11 = (row1 + x0 + 1) * 4;
      const pixDst = dstOff + dx * 4;

      for (let c = 0; c < 4; c++) {
        const top = rgba[c00 + c] + (rgba[c10 + c] - rgba[c00 + c]) * fx;
        const bottom = rgba[c01 + c] + (rgba[c11 + c] - rgba[c01 + c]) * fx;
        out[pixDst + c] = Math.round(top + (bottom - top) * fy);
      }
    }
  }

  return { data: out, width: dstW, height: dstH, homography: H };
}


/**
 * Robustly fits a line y = m*x + b (or x = m*y + b when vertical=true) using 2-pass trimmed least squares.
 */
function fitTrimmedLine(
  pts: Point2D[],
  vertical: boolean
): { m: number; b: number } | null {
  if (pts.length < 2) return null;

  const solve = (subset: Point2D[]): { m: number; b: number } | null => {
    if (subset.length < 2) return null;
    let sumU = 0;
    let sumV = 0;
    for (const p of subset) {
      sumU += vertical ? p.y : p.x;
      sumV += vertical ? p.x : p.y;
    }
    const meanU = sumU / subset.length;
    const meanV = sumV / subset.length;
    let num = 0;
    let den = 0;
    for (const p of subset) {
      const du = (vertical ? p.y : p.x) - meanU;
      const dv = (vertical ? p.x : p.y) - meanV;
      num += du * dv;
      den += du * du;
    }
    const m = den > 1e-4 ? num / den : 0;
    const b = meanV - m * meanU;
    return { m, b };
  };

  let line = solve(pts);
  if (!line || pts.length < 4) return line;

  // Drop top 25% highest residual outliers for clean edge locking
  const residuals = pts.map((p) => {
    const u = vertical ? p.y : p.x;
    const v = vertical ? p.x : p.y;
    return { p, err: Math.abs(v - (line!.m * u + line!.b)) };
  });
  residuals.sort((a, b) => a.err - b.err);
  const keepCount = Math.max(3, Math.floor(residuals.length * 0.75));
  const inliers = residuals.slice(0, keepCount).map((r) => r.p);
  return solve(inliers) ?? line;
}

/**
 * Intersects a nearly-horizontal line (y = mh * x + bh) with a nearly-vertical line (x = mv * y + bv).
 */
function intersectHV(
  horiz: { m: number; b: number },
  vert: { m: number; b: number }
): Point2D | null {
  const denom = 1 - vert.m * horiz.m;
  if (Math.abs(denom) < 1e-5) return null;
  const x = (vert.m * horiz.b + vert.b) / denom;
  const y = horiz.m * x + horiz.b;
  return { x, y };
}

/**
 * Detects an angled or perspective-distorted bright paper sheet against a darker background
 * by tracing outer boundary transitions across horizontal and vertical bands, fitting 4
 * robust page edges, and intersecting adjacent edges to recover true slanted corners.
 */
export function detectDocumentQuad(
  gray: Uint8ClampedArray,
  w: number,
  h: number
): QuadCorners | null {
  if (w < 140 || h < 100) return null;

  // Estimate bright paper intensity dynamically from interior pixels
  const centerVals: number[] = [];
  const yStart = Math.round(h * 0.25);
  const yEnd = Math.round(h * 0.75);
  const xStart = Math.round(w * 0.25);
  const xEnd = Math.round(w * 0.75);
  for (let y = yStart; y < yEnd; y += 4) {
    const rowOff = y * w;
    for (let x = xStart; x < xEnd; x += 4) {
      centerVals.push(gray[rowOff + x]);
    }
  }
  centerVals.sort((a, b) => a - b);
  const paperP80 = centerVals[Math.floor(centerVals.length * 0.8)] ?? 240;
  if (paperP80 < 160) return null;
  const paperThresh = Math.max(145, Math.min(205, paperP80 - 45));

  // Sample horizontal paper spans (for left & right edge points) across 9 y-bands
  const leftPts: Point2D[] = [];
  const rightPts: Point2D[] = [];
  for (let k = 1; k <= 9; k++) {
    const by = Math.round((h * (0.1 + 0.08 * k)));
    if (by < 0 || by >= h) continue;
    const rowOff = by * w;

    // Require contiguous >= 5 bright pixels so stray background specks are ignored
    let lx = -1;
    for (let x = 0; x <= w - 5; x++) {
      if (
        gray[rowOff + x] >= paperThresh &&
        gray[rowOff + x + 2] >= paperThresh &&
        gray[rowOff + x + 4] >= paperThresh
      ) {
        lx = x;
        break;
      }
    }

    let rx = -1;
    for (let x = w - 1; x >= 4; x--) {
      if (
        gray[rowOff + x] >= paperThresh &&
        gray[rowOff + x - 2] >= paperThresh &&
        gray[rowOff + x - 4] >= paperThresh
      ) {
        rx = x;
        break;
      }
    }

    if (lx !== -1 && rx !== -1 && rx - lx >= w * 0.28) {
      leftPts.push({ x: lx, y: by });
      rightPts.push({ x: rx, y: by });
    }
  }

  // Sample vertical paper spans (for top & bottom edge points) across 9 x-bands
  const topPts: Point2D[] = [];
  const bottomPts: Point2D[] = [];
  for (let k = 1; k <= 9; k++) {
    const bx = Math.round(w * (0.1 + 0.08 * k));
    if (bx < 0 || bx >= w) continue;

    let ty = -1;
    for (let y = 0; y <= h - 5; y++) {
      if (
        gray[y * w + bx] >= paperThresh &&
        gray[(y + 2) * w + bx] >= paperThresh &&
        gray[(y + 4) * w + bx] >= paperThresh
      ) {
        ty = y;
        break;
      }
    }

    let by = -1;
    for (let y = h - 1; y >= 4; y--) {
      if (
        gray[y * w + bx] >= paperThresh &&
        gray[(y - 2) * w + bx] >= paperThresh &&
        gray[(y - 4) * w + bx] >= paperThresh
      ) {
        by = y;
        break;
      }
    }

    if (ty !== -1 && by !== -1 && by - ty >= h * 0.28) {
      topPts.push({ x: bx, y: ty });
      bottomPts.push({ x: bx, y: by });
    }
  }

  if (leftPts.length < 4 || rightPts.length < 4 || topPts.length < 4 || bottomPts.length < 4) {
    return null;
  }

  // Verify dark border exists on at least 2 outer sides (avoids false positives on full-bleed pages)
  const avgLeft = leftPts.reduce((s, p) => s + p.x, 0) / leftPts.length;
  const avgRight = rightPts.reduce((s, p) => s + p.x, 0) / rightPts.length;
  const avgTop = topPts.reduce((s, p) => s + p.y, 0) / topPts.length;
  const avgBottom = bottomPts.reduce((s, p) => s + p.y, 0) / bottomPts.length;

  const hasBorders =
    (avgLeft > w * 0.025 ? 1 : 0) +
    (w - 1 - avgRight > w * 0.025 ? 1 : 0) +
    (avgTop > h * 0.025 ? 1 : 0) +
    (h - 1 - avgBottom > h * 0.025 ? 1 : 0);
  if (hasBorders < 2) return null;

  const leftLine = fitTrimmedLine(leftPts, true);
  const rightLine = fitTrimmedLine(rightPts, true);
  const topLine = fitTrimmedLine(topPts, false);
  const bottomLine = fitTrimmedLine(bottomPts, false);

  if (!leftLine || !rightLine || !topLine || !bottomLine) return null;

  const tl = intersectHV(topLine, leftLine);
  const tr = intersectHV(topLine, rightLine);
  const br = intersectHV(bottomLine, rightLine);
  const bl = intersectHV(bottomLine, leftLine);

  if (!tl || !tr || !br || !bl) return null;

  const clampPt = (p: Point2D): Point2D => ({
    x: Math.max(0, Math.min(w - 1, Math.round(p.x))),
    y: Math.max(0, Math.min(h - 1, Math.round(p.y))),
  });

  const quad = orderQuadCorners([clampPt(tl), clampPt(tr), clampPt(br), clampPt(bl)]);
  const [cTL, cTR, cBR, cBL] = quad;

  const spanW = Math.max(cTR.x - cTL.x, cBR.x - cBL.x);
  const spanH = Math.max(cBL.y - cTL.y, cBR.y - cTR.y);
  if (spanW < w * 0.42 || spanH < h * 0.42) return null;

  const quadArea =
    0.5 *
    Math.abs(
      cTL.x * cTR.y -
        cTR.x * cTL.y +
        (cTR.x * cBR.y - cBR.x * cTR.y) +
        (cBR.x * cBL.y - cBL.x * cBR.y) +
        (cBL.x * cTL.y - cTL.x * cBL.y)
    );
  if (quadArea < w * h * 0.25) return null;

  // Trigger only when genuine keystone taper or edge tilt (>= 2.5% of frame dimension) is present
  const leftSkew = Math.abs(cTL.x - cBL.x);
  const rightSkew = Math.abs(cTR.x - cBR.x);
  const topTilt = Math.abs(cTL.y - cTR.y);
  const botTilt = Math.abs(cBL.y - cBR.y);
  const widthTaper = Math.abs((cTR.x - cTL.x) - (cBR.x - cBL.x));
  const heightTaper = Math.abs((cBL.y - cTL.y) - (cBR.y - cTR.y));

  const maxDistortion = Math.max(leftSkew, rightSkew, topTilt, botTilt, widthTaper, heightTaper);
  if (maxDistortion < Math.min(w, h) * 0.025) {
    return null;
  }

  return quad;
}

/**
 * Estimates dominant text skew angle in degrees (default -18° to +18°) using a 2-pass
 * coarse-to-fine Radon horizontal projection variance profile over foreground ink pixels.
 *
 * Sign convention: positive angleDeg means text slopes upward-right and rotateGrayscale
 * rotates clockwise by angleDeg to level lines horizontally.
 */
export function estimateSkewAngle(
  binary: Uint8ClampedArray,
  w: number,
  h: number,
  options: DeskewOptions = {}
): number {
  if (w < 40 || h < 40) return 0;

  const maxAngle = options.maxAngleDeg ?? 28;
  const coarseStep = options.coarseStepDeg ?? 0.5;
  const fineStep = options.fineStepDeg ?? 0.1;

  const pointsX: number[] = [];
  const pointsY: number[] = [];
  const step = w * h > 400_000 ? 3 : w * h > 120_000 ? 2 : 1;
  const cx = w / 2;
  const cy = h / 2;

  for (let y = 0; y < h; y += step) {
    const rowOff = y * w;
    for (let x = 0; x < w; x += step) {
      if (binary[rowOff + x] === 0) {
        pointsX.push(x - cx);
        pointsY.push(y - cy);
      }
    }
  }

  if (pointsX.length < 30) return 0;

  // Use padded vertical bin span so tall rotated corners don't get clipped out of scoring
  const diag = Math.ceil(Math.hypot(w, h));
  const halfDiag = Math.ceil(diag / 2);
  const bins = new Int32Array(diag + 4);

  const scoreAngle = (deg: number): number => {
    const rad = (deg * Math.PI) / 180;
    const sin = Math.sin(rad);
    const cos = Math.cos(rad);
    bins.fill(0);

    for (let i = 0; i < pointsX.length; i++) {
      const ry = Math.round(pointsX[i] * sin + pointsY[i] * cos + halfDiag);
      if (ry >= 0 && ry < bins.length) {
        bins[ry]++;
      }
    }

    // Sum squared local 2-row energy + sharp row contrast to lock crisp horizontal baselines
    let energy = 0;
    for (let i = 0; i < bins.length - 1; i++) {
      const c = bins[i];
      if (c > 0) {
        energy += c * c + c * bins[i + 1] * 0.35;
      }
    }
    return energy;
  };

  let bestAngle = 0;
  let bestScore = scoreAngle(0);

  for (let deg = -maxAngle; deg <= maxAngle + 1e-6; deg += coarseStep) {
    const s = scoreAngle(deg);
    if (s > bestScore) {
      bestScore = s;
      bestAngle = deg;
    }
  }

  const fineStart = Math.max(-maxAngle, bestAngle - coarseStep * 1.5);
  const fineEnd = Math.min(maxAngle, bestAngle + coarseStep * 1.5);
  for (let deg = fineStart; deg <= fineEnd + 1e-6; deg += fineStep) {
    const s = scoreAngle(deg);
    if (s > bestScore) {
      bestScore = s;
      bestAngle = deg;
    }
  }

  return Number(bestAngle.toFixed(2));
}

/**
 * Rotates a grayscale buffer around its center by angleDeg using bilinear interpolation,
 * padding exposed corners cleanly with backgroundFill (default white paper 255).
 */
export function rotateGrayscale(
  gray: Uint8ClampedArray,
  w: number,
  h: number,
  angleDeg: number,
  backgroundFill = 255
): Uint8ClampedArray {
  if (Math.abs(angleDeg) < 0.05) return new Uint8ClampedArray(gray);

  const out = new Uint8ClampedArray(w * h);
  out.fill(backgroundFill);

  const rad = (-angleDeg * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const cx = (w - 1) / 2;
  const cy = (h - 1) / 2;

  for (let y = 0; y < h; y++) {
    const dy = y - cy;
    const rowOff = y * w;
    for (let x = 0; x < w; x++) {
      const dx = x - cx;
      const sx = dx * cos - dy * sin + cx;
      const sy = dx * sin + dy * cos + cy;

      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      if (x0 < 0 || x0 >= w - 1 || y0 < 0 || y0 >= h - 1) continue;

      const fx = sx - x0;
      const fy = sy - y0;
      const i00 = gray[y0 * w + x0];
      const i10 = gray[y0 * w + x0 + 1];
      const i01 = gray[(y0 + 1) * w + x0];
      const i11 = gray[(y0 + 1) * w + x0 + 1];

      const top = i00 + (i10 - i00) * fx;
      const bottom = i01 + (i11 - i01) * fx;
      out[rowOff + x] = Math.round(top + (bottom - top) * fy);
    }
  }

  return out;
}

/**
 * Convenience helper combining skew estimation on a binarized text mask with
 * bilinear rotation of the underlying grayscale document buffer.
 */
export function deskewGrayscale(
  gray: Uint8ClampedArray,
  binary: Uint8ClampedArray,
  w: number,
  h: number,
  options: DeskewOptions = {}
): DeskewResult {
  const minCorrection = options.minCorrectionDeg ?? 0.4;
  const angleDeg = estimateSkewAngle(binary, w, h, options);
  if (Math.abs(angleDeg) < minCorrection) {
    return { data: new Uint8ClampedArray(gray), angleDeg: 0, corrected: false };
  }
  return {
    data: rotateGrayscale(gray, w, h, angleDeg, options.backgroundFill ?? 255),
    angleDeg,
    corrected: true,
  };
}

/**
 * Convenience helper running automatic document quad detection + 3x3 projective homography
 * dewarping on a flattened grayscale buffer.
 */
export function correctDocumentPerspective(
  gray: Uint8ClampedArray,
  w: number,
  h: number,
  options: PerspectiveWarpOptions = {}
): PerspectiveResult {
  const quad = detectDocumentQuad(gray, w, h);
  if (!quad) {
    return {
      data: new Uint8ClampedArray(gray),
      width: w,
      height: h,
      quad: null,
      warped: false,
      homography: null,
    };
  }

  const warped = warpPerspectiveQuad(gray, w, h, quad, options);
  return {
    data: warped.data,
    width: warped.width,
    height: warped.height,
    quad,
    warped: true,
    homography: warped.homography,
  };
}

export interface DeslantOptions {
  maxSlantDeg?: number;
  stepDeg?: number;
  minCorrectionDeg?: number;
  inkThreshold?: number;
  backgroundFill?: number;
}

export interface DeslantResult {
  data: Uint8ClampedArray;
  width: number;
  height: number;
  slantDeg: number;
  corrected: boolean;
}

/**
 * Estimates cursive/handwritten slant angle (-25° to +25°) by maximizing vertical
 * column ink concentration under horizontal shear, then applies bilinear shear
 * deslanting so upright stems align vertically for TrOCR recognition.
 */
export function deslantGrayscale(
  gray: Uint8ClampedArray,
  w: number,
  h: number,
  options: DeslantOptions = {}
): DeslantResult {
  if (w < 8 || h < 8) {
    return {
      data: new Uint8ClampedArray(gray),
      width: w,
      height: h,
      slantDeg: 0,
      corrected: false,
    };
  }

  const maxSlant = options.maxSlantDeg ?? 25;
  const stepDeg = options.stepDeg ?? 2.5;
  const minCorrection = options.minCorrectionDeg ?? 3.0;
  const inkThreshold = options.inkThreshold ?? 175;
  const fill = options.backgroundFill ?? 255;

  const inkX: number[] = [];
  const inkY: number[] = [];
  const cy = (h - 1) * 0.5;

  for (let y = 0; y < h; y++) {
    const rowOff = y * w;
    for (let x = 0; x < w; x++) {
      if (gray[rowOff + x] < inkThreshold) {
        inkX.push(x);
        inkY.push(y - cy);
      }
    }
  }

  if (inkX.length < 20) {
    return {
      data: new Uint8ClampedArray(gray),
      width: w,
      height: h,
      slantDeg: 0,
      corrected: false,
    };
  }

  const maxExtraW = Math.ceil(Math.tan((maxSlant * Math.PI) / 180) * h);
  const binW = w + maxExtraW * 2 + 4;
  const colCounts = new Int32Array(binW);

  const scoreSlant = (deg: number): number => {
    const k = Math.tan((deg * Math.PI) / 180);
    colCounts.fill(0);
    for (let i = 0; i < inkX.length; i++) {
      const sx = Math.round(inkX[i] + k * inkY[i] + maxExtraW + 2);
      if (sx >= 0 && sx < binW) {
        colCounts[sx]++;
      }
    }
    let energy = 0;
    for (let x = 0; x < binW; x++) {
      const c = colCounts[x];
      if (c > 0) energy += c * c;
    }
    return energy;
  };

  const zeroEnergy = scoreSlant(0);
  let bestDeg = 0;
  let bestEnergy = zeroEnergy;

  if (options.stepDeg !== undefined) {
    for (let deg = -maxSlant; deg <= maxSlant + 1e-6; deg += stepDeg) {
      const energy = scoreSlant(deg);
      if (energy > bestEnergy) {
        bestEnergy = energy;
        bestDeg = deg;
      }
    }
  } else {
    // Coarse-to-fine sweep: step 8 deg coarse pass, then ±4 deg fine pass at 2 deg step
    let coarseBestDeg = 0;
    let coarseBestEnergy = zeroEnergy;
    const coarseLimit = Math.floor(maxSlant / 8) * 8;
    for (let deg = -coarseLimit; deg <= coarseLimit; deg += 8) {
      if (deg === 0) continue;
      const energy = scoreSlant(deg);
      if (energy > coarseBestEnergy) {
        coarseBestEnergy = energy;
        coarseBestDeg = deg;
      }
    }

    bestDeg = coarseBestDeg;
    bestEnergy = coarseBestEnergy;
    const fineStart = Math.max(-maxSlant, coarseBestDeg - 4);
    const fineEnd = Math.min(maxSlant, coarseBestDeg + 4);
    for (let deg = fineStart; deg <= fineEnd + 1e-6; deg += 2) {
      if (deg === coarseBestDeg || deg === 0) continue;
      const energy = scoreSlant(deg);
      if (energy > bestEnergy) {
        bestEnergy = energy;
        bestDeg = deg;
      }
    }
  }

  if (Math.abs(bestDeg) < minCorrection || bestEnergy <= zeroEnergy * 1.05) {
    return {
      data: new Uint8ClampedArray(gray),
      width: w,
      height: h,
      slantDeg: 0,
      corrected: false,
    };
  }

  const k = Math.tan((bestDeg * Math.PI) / 180);
  const padX = Math.ceil(Math.abs(k) * (h * 0.5)) + 2;
  const outW = w + padX * 2;
  const out = new Uint8ClampedArray(outW * h).fill(fill);

  for (let y = 0; y < h; y++) {
    const dy = y - cy;
    const dstOff = y * outW;
    const row0 = y * w;
    for (let dx = 0; dx < outW; dx++) {
      // Inverse mapping: dstX = srcX + k * (y - cy) + padX  =>  srcX = dstX - padX - k * dy
      const sx = dx - padX - k * dy;
      const x0 = Math.floor(sx);
      if (x0 < 0 || x0 >= w - 1) continue;
      const fx = sx - x0;
      const v0 = gray[row0 + x0];
      const v1 = gray[row0 + x0 + 1];
      out[dstOff + dx] = Math.round(v0 + (v1 - v0) * fx);
    }
  }

  return {
    data: out,
    width: outW,
    height: h,
    slantDeg: Number(bestDeg.toFixed(1)),
    corrected: true,
  };
}

