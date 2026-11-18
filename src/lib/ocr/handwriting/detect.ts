import { LineBox, segmentLines } from './lines';

export interface HandwritingDetectionResult {
  isHandwriting: boolean;
  score: number;
  metrics: {
    heightCv: number;
    baselineWobble: number;
    joinedRatio: number;
    strokeWidthCv?: number;
    gapCv?: number;
  };
}

export interface DetectHandwritingOptions {
  lines?: LineBox[];
  skipFilter?: boolean;
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

function pickRepresentativeLines(lines: LineBox[], maxSample: number): LineBox[] {
  if (lines.length <= maxSample) return lines;
  const picked: LineBox[] = [];
  const used = new Set<number>();
  for (let i = 0; i < maxSample; i++) {
    const idx = Math.round((i * (lines.length - 1)) / (maxSample - 1));
    if (!used.has(idx)) {
      used.add(idx);
      picked.push(lines[idx]);
    }
  }
  return picked;
}

/**
 * Fast (< 5ms) auto-mode classifier separating printed fonts from cursive and semi-disconnected
 * handwritten ink. Reuses pre-segmented lines when provided, samples representative lines
 * with early-exit, and combines component height variance, baseline/x-height jitter,
 * inter-component gap variance, stroke-width variance, and joined multi-letter aspect ratios.
 */
export function detectHandwriting(
  binary: Uint8ClampedArray,
  w: number,
  h: number,
  options?: DetectHandwritingOptions
): HandwritingDetectionResult {
  const lines =
    options?.lines && options.lines.length > 0
      ? options.lines
      : segmentLines(binary, w, h, { skipFilter: options?.skipFilter });

  if (lines.length === 0) {
    return {
      isHandwriting: false,
      score: 0,
      metrics: { heightCv: 0, baselineWobble: 0, joinedRatio: 0, strokeWidthCv: 0, gapCv: 0 },
    };
  }

  const allHeights: number[] = [];
  const lineWobbles: number[] = [];
  const strokeRuns: number[] = [];
  const interGaps: number[] = [];
  let wideCount = 0;
  let totalComps = 0;

  // Sample up to 6 representative lines (top, middle, bottom) first, then up to 12 only if borderline
  const primarySample = pickRepresentativeLines(lines, 6);
  const primarySet = new Set(primarySample);
  const remainingLines = lines.filter((l) => !primarySet.has(l)).slice(0, 6);

  const computeSnapshot = () => {
    if (totalComps === 0) {
      return {
        isHandwriting: false,
        score: 0,
        heightCv: 0,
        baselineWobble: 0,
        joinedRatio: 0,
        strokeWidthCv: 0,
        gapCv: 0,
      };
    }

    const meanAllH = allHeights.reduce((a, b) => a + b, 0) / allHeights.length;
    const heightCv = meanAllH > 0 ? stdDev(allHeights) / meanAllH : 0;
    const baselineWobble =
      lineWobbles.length > 0
        ? lineWobbles.reduce((a, b) => a + b, 0) / lineWobbles.length
        : 0;
    const joinedRatio = wideCount / totalComps;

    let strokeWidthCv = 0;
    if (strokeRuns.length >= 8) {
      const meanRun = strokeRuns.reduce((a, b) => a + b, 0) / strokeRuns.length;
      strokeWidthCv = meanRun > 0 ? stdDev(strokeRuns) / meanRun : 0;
    }

    let gapCv = 0;
    if (interGaps.length >= 4) {
      const meanGap = interGaps.reduce((a, b) => a + b, 0) / interGaps.length;
      gapCv = meanGap > 0 ? stdDev(interGaps) / meanGap : 0;
    }

    const score = Math.min(
      1,
      joinedRatio * 1.25 +
        Math.min(0.35, heightCv * 0.65) +
        Math.min(0.35, baselineWobble * 1.15) +
        Math.min(0.22, Math.max(0, strokeWidthCv - 0.18) * 0.45) +
        Math.min(0.1, Math.max(0, gapCv - 0.45) * 0.15)
    );

    const hasHumanPenmanship =
      (joinedRatio >= 0.12 && (heightCv >= 0.14 || baselineWobble >= 0.07)) ||
      (heightCv >= 0.18 && baselineWobble >= 0.09) ||
      (strokeWidthCv >= 0.38 && (heightCv >= 0.13 || baselineWobble >= 0.06)) ||
      (gapCv >= 0.55 && heightCv >= 0.15 && baselineWobble >= 0.07);

    return {
      isHandwriting: score >= 0.42 || (score >= 0.30 && hasHumanPenmanship),
      score,
      heightCv,
      baselineWobble,
      joinedRatio,
      strokeWidthCv,
      gapCv,
    };
  };

  const inspectLine = (box: LineBox) => {
    const rawComps = extractLineComponents(binary, w, box.y0, box.y1, box.x0, box.x1);
    const sized = rawComps.filter((c) => {
      const cw = c.x1 - c.x0 + 1;
      const ch = c.y1 - c.y0 + 1;
      return cw <= w * 0.48 && ch <= h * 0.3 && ch >= 5;
    });
    if (sized.length === 0) return;

    const sortedH = sized.map((c) => c.y1 - c.y0 + 1).sort((a, b) => a - b);
    const medianCompH = sortedH[Math.floor(sortedH.length * 0.5)] || 10;
    const comps = sized
      .filter((c) => c.y1 - c.y0 + 1 >= Math.max(6, medianCompH * 0.45))
      .sort((a, b) => a.x0 - b.x0);
    if (comps.length === 0) return;

    const lineHeights = comps.map((c) => c.y1 - c.y0 + 1);
    const meanH = lineHeights.reduce((a, b) => a + b, 0) / lineHeights.length;
    allHeights.push(...lineHeights);

    // Measure baseline wobble (bottom y1 + top y0 x-height jitter) around least-squares line slope
    if (meanH > 0 && comps.length >= 3) {
      const xs = comps.map((c) => (c.x0 + c.x1) * 0.5);
      const ys = comps.map((c) => c.y1);
      const y0s = comps.map((c) => c.y0);
      const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
      const my = ys.reduce((a, b) => a + b, 0) / ys.length;
      const my0 = y0s.reduce((a, b) => a + b, 0) / y0s.length;
      let num = 0;
      let num0 = 0;
      let den = 0;
      for (let i = 0; i < xs.length; i++) {
        const dx = xs[i] - mx;
        num += dx * (ys[i] - my);
        num0 += dx * (y0s[i] - my0);
        den += dx * dx;
      }
      const slope = den > 1e-3 ? num / den : 0;
      const slope0 = den > 1e-3 ? num0 / den : 0;
      const residuals = ys.map((y, i) => y - (my + slope * (xs[i] - mx)));
      const residuals0 = y0s.map((y, i) => y - (my0 + slope0 * (xs[i] - mx)));
      const baseWobble = stdDev(residuals) / meanH;
      const topWobble = stdDev(residuals0) / meanH;
      lineWobbles.push(baseWobble * 0.72 + topWobble * 0.28);
    }

    // Collect intra-line inter-component gaps (zero extra pixel scans)
    for (let i = 1; i < comps.length; i++) {
      const g = comps[i].x0 - comps[i - 1].x1;
      if (g > 0 && g <= meanH * 1.5 && interGaps.length < 200) {
        interGaps.push(g);
      }
    }

    // Sample horizontal stroke run widths across component mid-bands for stroke-width variance
    for (const c of comps) {
      const cw = c.x1 - c.x0 + 1;
      const ch = c.y1 - c.y0 + 1;
      totalComps++;
      if (cw / Math.max(1, ch) >= 1.65) {
        wideCount++;
      }

      if (strokeRuns.length < 320) {
        const yStep = Math.max(1, Math.floor(ch / 5));
        for (let ry = c.y0 + 1; ry <= c.y1 - 1; ry += yStep) {
          const rowOff = ry * w;
          let run = 0;
          for (let rx = c.x0; rx <= c.x1; rx++) {
            if (binary[rowOff + rx] === 0) {
              run++;
            } else if (run > 0) {
              if (run <= Math.max(2, Math.round(ch * 0.65))) {
                strokeRuns.push(run);
              }
              run = 0;
            }
          }
          if (run > 0 && run <= Math.max(2, Math.round(ch * 0.65))) {
            strokeRuns.push(run);
          }
        }
      }
    }
  };

  for (const box of primarySample) {
    inspectLine(box);
  }

  // Early-exit when representative sample is already decisive
  let snap = computeSnapshot();
  if (remainingLines.length > 0 && snap.score > 0.15 && snap.score < 0.65) {
    for (const box of remainingLines) {
      inspectLine(box);
    }
    snap = computeSnapshot();
  }

  return {
    isHandwriting: snap.isHandwriting,
    score: Number(snap.score.toFixed(3)),
    metrics: {
      heightCv: Number(snap.heightCv.toFixed(3)),
      baselineWobble: Number(snap.baselineWobble.toFixed(3)),
      joinedRatio: Number(snap.joinedRatio.toFixed(3)),
      strokeWidthCv: Number(snap.strokeWidthCv.toFixed(3)),
      gapCv: Number(snap.gapCv.toFixed(3)),
    },
  };
}

