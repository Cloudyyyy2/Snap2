import { describe, it, expect } from 'vitest';
import {
  orderQuadCorners,
  solveHomography,
  applyHomographyPoint,
  warpPerspectiveQuad,
  warpPerspectiveRgba,
  detectDocumentQuad,
  correctDocumentPerspective,
  estimateSkewAngle,
  rotateGrayscale,
  deskewGrayscale,
  estimateSlantAngle,
  estimateCursiveSlantAngle,
  shearDeslantGrayscale,
  shearGrayscale,
  deslantGrayscale,
  type SlantOptions,
  type SlantResult,
  type QuadCorners,
} from '../../src/lib/ocr/geometry';
import {
  estimateSlantAngle as cleanupEstimateSlantAngle,
  shearDeslantGrayscale as cleanupShearDeslantGrayscale,
  deslantGrayscale as cleanupDeslantGrayscale,
} from '../../src/lib/ocr/handwriting/cleanup';

function renderTiltedTextLines(
  w: number,
  h: number,
  tiltDeg: number
): Uint8ClampedArray {
  const buf = new Uint8ClampedArray(w * h).fill(255);
  const rad = (tiltDeg * Math.PI) / 180;
  const tan = Math.tan(rad);
  const cx = w / 2;

  // Draw 4 crisp horizontal baselines tilted by tiltDeg (positive = slopes upward-right)
  const baseYs = [Math.round(h * 0.22), Math.round(h * 0.42), Math.round(h * 0.62), Math.round(h * 0.8)];
  for (const by of baseYs) {
    for (let x = 40; x < w - 40; x++) {
      // Simulate letter gaps so it looks like real words
      if (x % 22 >= 16) continue;
      const centerY = by - (x - cx) * tan;
      for (let dy = -4; dy <= 4; dy++) {
        const py = Math.round(centerY + dy);
        if (py >= 0 && py < h) {
          buf[py * w + x] = 0;
        }
      }
    }
  }
  return buf;
}

describe('geometric & spatial correction (deskewing & perspective dewarping)', () => {
  it('orderQuadCorners sorts shuffled points consistently to [TL, TR, BR, BL]', () => {
    const tl = { x: 42, y: 28 };
    const tr = { x: 290, y: 40 };
    const br = { x: 315, y: 210 };
    const bl = { x: 18, y: 195 };

    const ordered = orderQuadCorners([br, tl, bl, tr]);
    expect(ordered).toEqual([tl, tr, br, bl]);
  });

  it('solveHomography and applyHomographyPoint map 4 projective corners accurately', () => {
    const src: QuadCorners = [
      { x: 0, y: 0 },
      { x: 200, y: 0 },
      { x: 200, y: 150 },
      { x: 0, y: 150 },
    ];
    const dst: QuadCorners = [
      { x: 55, y: 24 },
      { x: 225, y: 38 },
      { x: 268, y: 195 },
      { x: 22, y: 182 },
    ];

    const H = solveHomography(src, dst);
    expect(H).not.toBeNull();
    expect(H!.length).toBe(9);

    for (let i = 0; i < 4; i++) {
      const mapped = applyHomographyPoint(H!, src[i].x, src[i].y);
      expect(Math.abs(mapped.x - dst[i].x)).toBeLessThan(0.01);
      expect(Math.abs(mapped.y - dst[i].y)).toBeLessThan(0.01);
    }
  });

  it('warpPerspectiveQuad flattens a projective keystone document and preserves level text rows', () => {
    const srcW = 360;
    const srcH = 260;
    const docW = 240;
    const docH = 180;

    // Create flat upright paper document with dark horizontal text bars
    const flatDoc = new Uint8ClampedArray(docW * docH).fill(242);
    for (const rowY of [40, 90, 140]) {
      for (let y = rowY - 4; y <= rowY + 4; y++) {
        for (let x = 30; x < docW - 30; x++) {
          flatDoc[y * docW + x] = 20;
        }
      }
    }

    // Forward-project flatDoc into a 3D keystone quad on a dark desk
    const destQuad: QuadCorners = [
      { x: 75, y: 35 },
      { x: 270, y: 22 },
      { x: 328, y: 228 },
      { x: 32, y: 215 },
    ];
    const invRect: QuadCorners = [
      { x: 0, y: 0 },
      { x: docW - 1, y: 0 },
      { x: docW - 1, y: docH - 1 },
      { x: 0, y: docH - 1 },
    ];

    const H_quadToDoc = solveHomography(destQuad, invRect)!;
    const cameraFrame = new Uint8ClampedArray(srcW * srcH).fill(32); // dark desk

    for (let y = 0; y < srcH; y++) {
      for (let x = 0; x < srcW; x++) {
        const pt = applyHomographyPoint(H_quadToDoc, x, y);
        if (pt.x >= 0 && pt.x < docW && pt.y >= 0 && pt.y < docH) {
          cameraFrame[y * srcW + x] = flatDoc[Math.round(pt.y) * docW + Math.round(pt.x)];
        }
      }
    }

    const res = warpPerspectiveQuad(cameraFrame, srcW, srcH, destQuad);
    expect(res.homography).not.toBeNull();
    expect(res.width).toBeGreaterThanOrEqual(280);
    expect(res.height).toBeGreaterThanOrEqual(190);

    // Corners of the warped sheet must be bright paper (not dark desk)
    expect(res.data[8 * res.width + 8]).toBeGreaterThan(210);
    expect(res.data[8 * res.width + (res.width - 9)]).toBeGreaterThan(210);
    expect(res.data[(res.height - 9) * res.width + 8]).toBeGreaterThan(210);
    expect(res.data[(res.height - 9) * res.width + (res.width - 9)]).toBeGreaterThan(210);

    // Text bands inside the dewarped sheet must be horizontally flat (skew ~ 0)
    const warpedBin = new Uint8ClampedArray(res.data.length);
    for (let i = 0; i < res.data.length; i++) {
      warpedBin[i] = res.data[i] < 130 ? 0 : 255;
    }
    const residualSkew = estimateSkewAngle(warpedBin, res.width, res.height);
    expect(Math.abs(residualSkew)).toBeLessThan(0.6);
  });

  it('detectDocumentQuad and correctDocumentPerspective automatically detect & rectify an angled sheet on a dark desk, and skip flat full-bleed pages', () => {
    const w = 360;
    const h = 260;

    // 1. Angled trapezoidal bright sheet on a dark desk
    const angled = new Uint8ClampedArray(w * h).fill(35);
    const tl = { x: 80, y: 30 };
    const tr = { x: 280, y: 42 };
    const br = { x: 332, y: 226 };
    const bl = { x: 28, y: 214 };

    for (let y = 30; y <= 226; y++) {
      const vL = (y - tl.y) / Math.max(1, bl.y - tl.y);
      const vR = (y - tr.y) / Math.max(1, br.y - tr.y);
      const lx = Math.round(tl.x + (bl.x - tl.x) * Math.max(0, Math.min(1, vL)));
      const rx = Math.round(tr.x + (br.x - tr.x) * Math.max(0, Math.min(1, vR)));
      for (let x = lx; x <= rx; x++) {
        // Slanted top edge clipping
        const u = (x - lx) / Math.max(1, rx - lx);
        const topY = tl.y + (tr.y - tl.y) * u;
        const botY = bl.y + (br.y - bl.y) * u;
        if (y >= topY && y <= botY) {
          angled[y * w + x] = 240;
        }
      }
    }

    const detected = detectDocumentQuad(angled, w, h);
    expect(detected).not.toBeNull();
    expect(Math.abs(detected![0].x - tl.x)).toBeLessThanOrEqual(8);
    expect(Math.abs(detected![1].x - tr.x)).toBeLessThanOrEqual(8);
    expect(Math.abs(detected![2].x - br.x)).toBeLessThanOrEqual(8);
    expect(Math.abs(detected![3].x - bl.x)).toBeLessThanOrEqual(8);

    const persp = correctDocumentPerspective(angled, w, h);
    expect(persp.warped).toBe(true);
    expect(persp.homography).not.toBeNull();
    // Corners of rectified image should now be bright white paper rather than dark desk
    expect(persp.data[10 * persp.width + 10]).toBeGreaterThan(215);
    expect(persp.data[(persp.height - 10) * persp.width + (persp.width - 10)]).toBeGreaterThan(215);

    // 2. Full-bleed flat white page should NOT trigger unnecessary warping
    const flatFullBleed = new Uint8ClampedArray(w * h).fill(245);
    expect(detectDocumentQuad(flatFullBleed, w, h)).toBeNull();
    expect(correctDocumentPerspective(flatFullBleed, w, h).warped).toBe(false);
  });

  it('estimateSkewAngle, rotateGrayscale, and deskewGrayscale recover and level both +6.5° and -7.5° tilted text lines', () => {
    const w = 360;
    const h = 220;

    for (const targetTilt of [6.5, -7.5]) {
      const skewed = renderTiltedTextLines(w, h, targetTilt);
      const estimated = estimateSkewAngle(skewed, w, h);
      expect(Math.sign(estimated)).toBe(Math.sign(targetTilt));
      expect(Math.abs(estimated - targetTilt)).toBeLessThanOrEqual(0.6);

      const deskewed = deskewGrayscale(skewed, skewed, w, h);
      expect(deskewed.corrected).toBe(true);
      expect(Math.abs(deskewed.angleDeg - targetTilt)).toBeLessThanOrEqual(0.6);

      // Re-binarize leveled output cleanly and confirm residual skew is < 0.5°
      const leveledBin = new Uint8ClampedArray(deskewed.data.length);
      for (let i = 0; i < deskewed.data.length; i++) {
        leveledBin[i] = deskewed.data[i] < 140 ? 0 : 255;
      }
      const residual = estimateSkewAngle(leveledBin, w, h);
      expect(Math.abs(residual)).toBeLessThan(0.5);
    }

    // Flat 0° image should not be needlessly rotated
    const flat = renderTiltedTextLines(w, h, 0);
    const flatRes = deskewGrayscale(flat, flat, w, h);
    expect(flatRes.corrected).toBe(false);
    expect(flatRes.angleDeg).toBe(0);

    // Direct rotateGrayscale smoke test
    const rotated = rotateGrayscale(flat, w, h, 5);
    expect(rotated.length).toBe(flat.length);
    expect(rotated[0]).toBe(255); // exposed corner filled cleanly with paper white
  });

  it('warpPerspectiveRgba rectifies a 4-channel RGBA trapezoid without color distortion or NaN values', () => {
    const srcW = 300;
    const srcH = 200;
    // Dark desk: RGBA (35, 30, 25, 255)
    const rgba = new Uint8ClampedArray(srcW * srcH * 4);
    for (let i = 0; i < srcW * srcH; i++) {
      rgba[i * 4] = 35;
      rgba[i * 4 + 1] = 30;
      rgba[i * 4 + 2] = 25;
      rgba[i * 4 + 3] = 255;
    }

    // Yellow sticky note quadrilateral: RGBA (254, 240, 138, 255)
    const quad: QuadCorners = [
      { x: 50, y: 30 },
      { x: 250, y: 45 },
      { x: 280, y: 175 },
      { x: 30, y: 160 },
    ];

    // Paint interior of the trapezoid yellow
    const [tl, tr, br, bl] = quad;
    for (let y = 30; y <= 175; y++) {
      const vL = (y - tl.y) / Math.max(1, bl.y - tl.y);
      const vR = (y - tr.y) / Math.max(1, br.y - tr.y);
      const lx = Math.round(tl.x + (bl.x - tl.x) * Math.max(0, Math.min(1, vL)));
      const rx = Math.round(tr.x + (br.x - tr.x) * Math.max(0, Math.min(1, vR)));
      for (let x = lx; x <= rx; x++) {
        const u = (x - lx) / Math.max(1, rx - lx);
        const topY = tl.y + (tr.y - tl.y) * u;
        const botY = bl.y + (br.y - bl.y) * u;
        if (y >= topY && y <= botY) {
          const off = (y * srcW + x) * 4;
          rgba[off] = 254; // Red
          rgba[off + 1] = 240; // Green
          rgba[off + 2] = 138; // Blue
          rgba[off + 3] = 255; // Alpha
        }
      }
    }

    const warped = warpPerspectiveRgba(rgba, srcW, srcH, quad, {
      targetWidth: 200,
      targetHeight: 140,
    });

    expect(warped.width).toBe(200);
    expect(warped.height).toBe(140);
    expect(warped.data.length).toBe(200 * 140 * 4);
    expect(warped.homography).not.toBeNull();

    // Verify interior pixels are yellow sticky note (not dark desk)
    const sampleX = 100;
    const sampleY = 70;
    const sampleOff = (sampleY * warped.width + sampleX) * 4;
    expect(warped.data[sampleOff]).toBeGreaterThan(240); // R
    expect(warped.data[sampleOff + 1]).toBeGreaterThan(220); // G
    expect(warped.data[sampleOff + 2]).toBeLessThan(160); // B
    expect(warped.data[sampleOff + 3]).toBe(255); // A

    // Ensure no NaNs or uninitialized channels in output buffer
    for (let i = 0; i < warped.data.length; i += 4) {
      expect(Number.isNaN(warped.data[i])).toBe(false);
      expect(Number.isNaN(warped.data[i + 1])).toBe(false);
      expect(Number.isNaN(warped.data[i + 2])).toBe(false);
      expect(warped.data[i + 3]).toBe(255);
    }
  });

  it('estimateCursiveSlantAngle, shearGrayscale, and deslantGrayscale normalize slanted cursive strokes without clipping', () => {
    const w = 240;
    const h = 70;

    function renderSlantedStems(angleDeg: number): Uint8ClampedArray {
      const buf = new Uint8ClampedArray(w * h).fill(255);
      const rad = (angleDeg * Math.PI) / 180;
      const tan = Math.tan(rad);
      const stemX = [40, 65, 90, 115, 140, 165, 190];

      for (const sx of stemX) {
        for (let y = 12; y <= 58; y++) {
          const xCenter = sx + tan * (h - 1 - y);
          for (let dx = -1; dx <= 1; dx++) {
            const px = Math.round(xCenter + dx);
            if (px >= 0 && px < w) {
              buf[y * w + px] = 0;
            }
          }
        }
      }
      return buf;
    }

    function projectionVariance(buf: Uint8ClampedArray, width: number, height: number): number {
      const colSum = new Float64Array(width);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          if (buf[y * width + x] < 128) colSum[x]++;
        }
      }
      const mean = colSum.reduce((a, b) => a + b, 0) / width;
      return colSum.reduce((a, b) => a + (b - mean) ** 2, 0) / width;
    }

    // 1. Forward slant (+22°)
    const slanted22 = renderSlantedStems(22);
    const estimated22 = estimateCursiveSlantAngle(slanted22, w, h);
    expect(Math.abs(estimated22 - 22)).toBeLessThanOrEqual(1.5);

    const deslanted22 = deslantGrayscale(slanted22, slanted22, w, h);
    expect(deslanted22.deslanted).toBe(true);
    expect(deslanted22.width).toBeGreaterThanOrEqual(w);
    expect(deslanted22.height).toBe(h);

    const varOrig = projectionVariance(slanted22, w, h);
    const varDeslanted = projectionVariance(deslanted22.data, deslanted22.width, deslanted22.height);
    expect(varDeslanted).toBeGreaterThan(varOrig * 1.35);

    // Verify no stroke pixels are clipped: total ink should be conserved
    let inkOrig = 0;
    for (let i = 0; i < slanted22.length; i++) if (slanted22[i] === 0) inkOrig++;
    let inkDeslanted = 0;
    for (let i = 0; i < deslanted22.data.length; i++) if (deslanted22.data[i] < 128) inkDeslanted++;
    expect(Math.abs(inkDeslanted - inkOrig) / inkOrig).toBeLessThan(0.05);

    // 2. Upright text (0° slant) should not be unnecessarily sheared
    const upright = renderSlantedStems(0);
    const estimated0 = estimateCursiveSlantAngle(upright, w, h);
    expect(Math.abs(estimated0)).toBeLessThan(2.5);
    const deslanted0 = deslantGrayscale(upright, upright, w, h);
    expect(deslanted0.deslanted).toBe(false);
    expect(deslanted0.angleDeg).toBe(0);

    // 3. Backward slant (-18°)
    const slantedNeg = renderSlantedStems(-18);
    const estimatedNeg = estimateCursiveSlantAngle(slantedNeg, w, h);
    expect(Math.abs(estimatedNeg - -18)).toBeLessThanOrEqual(1.5);
    const deslantedNeg = deslantGrayscale(slantedNeg, slantedNeg, w, h);
    expect(deslantedNeg.deslanted).toBe(true);
  });

  it('F4 contract: handles strong ±34° slants, cleanup.ts re-exports, zero edge clipping at x=0/x=w-1, upright printed text, and degenerate buffers', () => {
    // 1. Contract & re-export identity checks
    expect(cleanupEstimateSlantAngle).toBe(estimateSlantAngle);
    expect(cleanupShearDeslantGrayscale).toBe(shearDeslantGrayscale);
    expect(cleanupDeslantGrayscale).toBe(deslantGrayscale);
    expect(estimateCursiveSlantAngle).toBe(estimateSlantAngle);
    expect(shearGrayscale).toBe(shearDeslantGrayscale);

    const w = 260;
    const h = 72;
    const renderStems = (angleDeg: number, stemXs: number[]): Uint8ClampedArray => {
      const buf = new Uint8ClampedArray(w * h).fill(255);
      const tan = Math.tan((angleDeg * Math.PI) / 180);
      const cy = (h - 1) * 0.5;
      for (const sx of stemXs) {
        for (let y = 10; y <= h - 11; y++) {
          const xCenter = sx - tan * (y - cy);
          for (let dx = -1; dx <= 1; dx++) {
            const px = Math.round(xCenter + dx);
            if (px >= 0 && px < w) {
              buf[y * w + px] = 0;
            }
          }
        }
      }
      return buf;
    };

    // 2. Strong forward (+34°) and backward (-32°) slants within ±40° range
    const opts: SlantOptions = { minAngleDeg: -40, maxAngleDeg: 40, coarseStepDeg: 2, fineStepDeg: 0.5 };
    const slanted34 = renderStems(34, [60, 90, 120, 150, 180, 210]);
    const est34 = estimateSlantAngle(slanted34, w, h, opts);
    expect(Math.abs(est34 - 34)).toBeLessThanOrEqual(1.5);
    const res34: SlantResult = deslantGrayscale(slanted34, slanted34, w, h, opts);
    expect(res34.applied).toBe(true);
    expect(res34.shearX).toBeGreaterThan(0.5);
    expect(res34.width).toBeGreaterThanOrEqual(w + Math.ceil(h * Math.abs(Math.tan((est34 * Math.PI) / 180))));

    const slantedNeg32 = renderStems(-32, [60, 90, 120, 150, 180, 210]);
    const estNeg32 = estimateSlantAngle(slantedNeg32, w, h, opts);
    expect(Math.abs(estNeg32 - -32)).toBeLessThanOrEqual(1.5);
    const resNeg32 = deslantGrayscale(slantedNeg32, slantedNeg32, w, h, opts);
    expect(resNeg32.applied).toBe(true);
    expect(resNeg32.shearX).toBeLessThan(-0.5);

    // 3. Edge-near characters touching x = 0 and x = w - 1 never clip at +30° or -30°
    const edgeW = 120;
    const edgeH = 60;
    for (const angle of [30, -30]) {
      const edgeImg = new Uint8ClampedArray(edgeW * edgeH).fill(255);
      // Draw vertical bars directly on leftmost (x=0..1) and rightmost (x=edgeW-2..edgeW-1) columns
      for (let y = 8; y < edgeH - 8; y++) {
        edgeImg[y * edgeW + 0] = 0;
        edgeImg[y * edgeW + 1] = 0;
        edgeImg[y * edgeW + (edgeW - 2)] = 0;
        edgeImg[y * edgeW + (edgeW - 1)] = 0;
      }
      const origInk = 2 * 2 * (edgeH - 16);
      const sheared = shearDeslantGrayscale(edgeImg, edgeW, edgeH, angle, 240);
      const minExpectedW = edgeW + Math.ceil(edgeH * Math.abs(Math.tan((angle * Math.PI) / 180)));
      expect(sheared.width).toBeGreaterThanOrEqual(minExpectedW);
      expect(sheared.height).toBe(edgeH);
      // Corners filled with custom backgroundFill (240)
      expect(sheared.data[0]).toBe(240);
      expect(sheared.data[sheared.width - 1]).toBe(240);

      let darkInk = 0;
      for (let i = 0; i < sheared.data.length; i++) {
        if (sheared.data[i] < 120) darkInk++;
      }
      expect(darkInk).toBeGreaterThanOrEqual(origInk - 4);
    }

    // 4. Upright printed letters + horizontal rule bars remain completely untouched
    const printed = renderStems(0, [40, 70, 100, 130, 160, 190]);
    for (let x = 35; x <= 195; x++) {
      printed[35 * w + x] = 0; // horizontal crossbar
      printed[36 * w + x] = 0;
    }
    expect(estimateSlantAngle(printed, w, h)).toBe(0);
    const resPrinted = deslantGrayscale(printed, printed, w, h);
    expect(resPrinted.applied).toBe(false);
    expect(resPrinted.shearX).toBe(0);
    expect(resPrinted.width).toBe(w);
    expect(resPrinted.data).toEqual(printed);

    // 5. Empty, all-white, all-black, and tiny (< 8x8) inputs are safe
    expect(estimateSlantAngle(new Uint8ClampedArray(0), 0, 0)).toBe(0);
    expect(estimateSlantAngle(new Uint8ClampedArray(4 * 4).fill(0), 4, 4)).toBe(0);
    expect(estimateSlantAngle(new Uint8ClampedArray(80 * 40).fill(255), 80, 40)).toBe(0);
    expect(estimateSlantAngle(new Uint8ClampedArray(80 * 40).fill(0), 80, 40)).toBe(0);

    const tinyRes = deslantGrayscale(new Uint8ClampedArray(5 * 5).fill(120), 5, 5);
    expect(tinyRes.applied).toBe(false);
    expect(tinyRes.shearX).toBe(0);
    expect(tinyRes.width).toBe(5);
    expect(tinyRes.height).toBe(5);
  });

  it('estimateSlantAngle, shearDeslantGrayscale, and deslantGrayscale safely handle NaN, Infinity, ±90°, and NaN backgroundFill', () => {
    const w = 40;
    const h = 30;
    const buf = new Uint8ClampedArray(w * h).fill(255);
    for (let y = 4; y < h - 4; y++) {
      buf[y * w + 15] = 0;
      buf[y * w + 16] = 0;
      buf[y * w + 25] = 0;
      buf[y * w + 26] = 0;
    }

    // 1. shearDeslantGrayscale with NaN / Infinity returns a clean copy with finite dims
    for (const badAngle of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      const res = shearDeslantGrayscale(buf, w, h, badAngle);
      expect(Number.isFinite(res.width)).toBe(true);
      expect(res.width).toBe(w);
      expect(res.height).toBe(h);
      expect(res.data.length).toBe(w * h);
      expect(res.data).toEqual(buf);
    }

    // 2. shearDeslantGrayscale with ±90° clamps safely instead of throwing RangeError
    for (const extremeAngle of [90, -90, 89.99, -89.99]) {
      const res = shearDeslantGrayscale(buf, w, h, extremeAngle);
      expect(Number.isFinite(res.width)).toBe(true);
      expect(res.width).toBeGreaterThanOrEqual(w);
      expect(res.height).toBe(h);
      expect(res.data.length).toBe(res.width * h);
    }

    // 3. shearDeslantGrayscale with backgroundFill = NaN falls back to 255 (white)
    const nanBgRes = shearDeslantGrayscale(buf, w, h, 20, Number.NaN);
    expect(nanBgRes.data[0]).toBe(255);
    expect(nanBgRes.data[nanBgRes.width - 1]).toBe(255);

    // 4. estimateSlantAngle with ±90° bounds or NaN/Infinity options does not throw
    expect(() =>
      estimateSlantAngle(buf, w, h, { minAngleDeg: -90, maxAngleDeg: 90 })
    ).not.toThrow();
    expect(
      Number.isFinite(
        estimateSlantAngle(buf, w, h, {
          minAngleDeg: Number.NaN,
          maxAngleDeg: Number.POSITIVE_INFINITY,
          coarseStepDeg: Number.NaN,
          fineStepDeg: Number.NEGATIVE_INFINITY,
        })
      )
    ).toBe(true);

    // 5. deslantGrayscale with NaN/Infinity dims returns safe empty result
    const nanDims = deslantGrayscale(buf, Number.NaN, h);
    expect(nanDims.applied).toBe(false);
    expect(nanDims.width).toBe(0);
    expect(nanDims.data.length).toBe(0);
  });
});
