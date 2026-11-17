import { describe, it, expect } from 'vitest';
import {
  orderQuadCorners,
  solveHomography,
  applyHomographyPoint,
  warpPerspectiveQuad,
  detectDocumentQuad,
  correctDocumentPerspective,
  estimateSkewAngle,
  rotateGrayscale,
  deskewGrayscale,
  type QuadCorners,
} from '../../src/lib/ocr/geometry';

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
});
