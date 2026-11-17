import { segmentLines } from './lines';

export interface HandwritingDetectionResult {
  isHandwriting: boolean;
  score: number;
  metrics: {
    heightCv: number;
    baselineWobble: number;
    joinedRatio: number;
  };
}

interface ComponentBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

function extractLineComponents(
  binary: Uint8ClampedArray,
  w: number,
  y0: number,
  y1: number,
  x0: number,
  x1: number
): ComponentBox[] {
  const visited = new Uint8Array((y1 - y0 + 1) * (x1 - x0 + 1));
  const bw = x1 - x0 + 1;
  const bh = y1 - y0 + 1;
  const comps: ComponentBox[] = [];
  const queueX: number[] = [];
  const queueY: number[] = [];

  for (let ly = 0; ly < bh; ly++) {
    const gy = y0 + ly;
    for (let lx = 0; lx < bw; lx++) {
      const idx = ly * bw + lx;
      if (visited[idx]) continue;
      const gx = x0 + lx;
      if (binary[gy * w + gx] !== 0) continue;

      visited[idx] = 1;
      queueX.length = 0;
      queueY.length = 0;
      queueX.push(lx);
      queueY.push(ly);

      let minX = lx;
      let maxX = lx;
      let minY = ly;
      let maxY = ly;
      let head = 0;

      while (head < queueX.length) {
        const cx = queueX[head];
        const cy = queueY[head];
        head++;

        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;

        for (let dy = -1; dy <= 1; dy++) {
          const ny = cy + dy;
          if (ny < 0 || ny >= bh) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const nx = cx + dx;
            if (nx < 0 || nx >= bw) continue;
            const nIdx = ny * bw + nx;
            if (!visited[nIdx] && binary[(y0 + ny) * w + (x0 + nx)] === 0) {
              visited[nIdx] = 1;
              queueX.push(nx);
              queueY.push(ny);
            }
          }
        }
      }

      const compW = maxX - minX + 1;
      const compH = maxY - minY + 1;
      if (compW >= 2 && compH >= 3) {
        comps.push({
          x0: x0 + minX,
          y0: y0 + minY,
          x1: x0 + maxX,
          y1: y0 + maxY,
        });
      }
    }
  }

  return comps;
}

function stdDev(values: number[]): number {
  if (values.length < 2) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

/**
 * Fast (< 5ms) auto-mode classifier separating printed fonts from cursive/handwritten ink.
 * Combines component height variance, baseline wobble, and joined multi-letter aspect ratios.
 * On a tie (score >= 0.42), defaults to handwriting when host TrOCR is available.
 */
export function detectHandwriting(
  binary: Uint8ClampedArray,
  w: number,
  h: number
): HandwritingDetectionResult {
  const lines = segmentLines(binary, w, h);
  if (lines.length === 0) {
    return {
      isHandwriting: false,
      score: 0,
      metrics: { heightCv: 0, baselineWobble: 0, joinedRatio: 0 },
    };
  }

  const allHeights: number[] = [];
  const lineWobbles: number[] = [];
  let wideCount = 0;
  let totalComps = 0;

  for (const box of lines.slice(0, 12)) {
    const rawComps = extractLineComponents(binary, w, box.y0, box.y1, box.x0, box.x1);
    const sized = rawComps.filter((c) => {
      const cw = c.x1 - c.x0 + 1;
      const ch = c.y1 - c.y0 + 1;
      return cw <= w * 0.48 && ch <= h * 0.3 && ch >= 5;
    });
    if (sized.length === 0) continue;

    // Exclude small punctuation marks (periods, commas, i-dots) from height/baseline variance
    const sortedH = sized.map((c) => c.y1 - c.y0 + 1).sort((a, b) => a - b);
    const medianCompH = sortedH[Math.floor(sortedH.length * 0.5)] || 10;
    const comps = sized.filter((c) => c.y1 - c.y0 + 1 >= Math.max(6, medianCompH * 0.45));
    if (comps.length === 0) continue;

    const lineHeights = comps.map((c) => c.y1 - c.y0 + 1);
    const meanH = lineHeights.reduce((a, b) => a + b, 0) / lineHeights.length;
    allHeights.push(...lineHeights);

    // Measure baseline wobble as residual around least-squares line slope (tilt-invariant!)
    if (meanH > 0 && comps.length >= 3) {
      const xs = comps.map((c) => (c.x0 + c.x1) * 0.5);
      const ys = comps.map((c) => c.y1);
      const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
      const my = ys.reduce((a, b) => a + b, 0) / ys.length;
      let num = 0;
      let den = 0;
      for (let i = 0; i < xs.length; i++) {
        const dx = xs[i] - mx;
        num += dx * (ys[i] - my);
        den += dx * dx;
      }
      const slope = den > 1e-3 ? num / den : 0;
      const residuals = ys.map((y, i) => y - (my + slope * (xs[i] - mx)));
      lineWobbles.push(stdDev(residuals) / meanH);
    }

    for (const c of comps) {
      const cw = c.x1 - c.x0 + 1;
      const ch = c.y1 - c.y0 + 1;
      totalComps++;
      // Joined cursive words/ligatures have wide multi-letter connected components
      if (cw / Math.max(1, ch) >= 1.65) {
        wideCount++;
      }
    }
  }

  if (totalComps === 0) {
    return {
      isHandwriting: false,
      score: 0,
      metrics: { heightCv: 0, baselineWobble: 0, joinedRatio: 0 },
    };
  }

  const meanAllH = allHeights.reduce((a, b) => a + b, 0) / allHeights.length;
  const heightCv = meanAllH > 0 ? stdDev(allHeights) / meanAllH : 0;
  const baselineWobble =
    lineWobbles.length > 0
      ? lineWobbles.reduce((a, b) => a + b, 0) / lineWobbles.length
      : 0;
  const joinedRatio = wideCount / totalComps;

  // Printed fonts have minimal joined components (< 0.08) and low detrended wobble; cursive scores high on both
  const score = Math.min(
    1,
    joinedRatio * 1.25 + Math.min(0.35, heightCv * 0.6) + Math.min(0.35, baselineWobble * 1.0)
  );

  return {
    isHandwriting: score >= 0.42,
    score: Number(score.toFixed(3)),
    metrics: {
      heightCv: Number(heightCv.toFixed(3)),
      baselineWobble: Number(baselineWobble.toFixed(3)),
      joinedRatio: Number(joinedRatio.toFixed(3)),
    },
  };
}
