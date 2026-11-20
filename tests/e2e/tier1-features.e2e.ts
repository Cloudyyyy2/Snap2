import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { preprocessForOCR } from '../../src/services/imageProcessor';
import { preprocessImageData, binarizeSauvola } from '../../src/lib/ocr/preprocess';
import { segmentLines, cropLine, splitWideLineBox } from '../../src/lib/ocr/handwriting/lines';
import { filterTextComponents } from '../../src/lib/ocr/handwriting/cleanup';
import {
  assert,
  assertEqual,
  assertGreaterOrEqual,
  assertIncludes,
  assertLessOrEqual,
  assertMatches,
  assertNotIncludes,
  characterAccuracy,
  loadOptionalModule,
  resolveExportFn,
} from './helpers/metrics';
import type { TestCase, TestContext } from './helpers/types';

async function pngToGrayBuffer(filePath: string): Promise<{ data: Uint8ClampedArray; width: number; height: number }> {
  const { data, info } = await sharp(filePath).grayscale().raw().toBuffer({ resolveWithObject: true });
  return {
    data: new Uint8ClampedArray(data),
    width: info.width,
    height: info.height,
  };
}

async function pngToRgbaImageData(filePath: string): Promise<{ data: Uint8ClampedArray; width: number; height: number }> {
  const { data, info } = await sharp(filePath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return {
    data: new Uint8ClampedArray(data),
    width: info.width,
    height: info.height,
  };
}

async function pngToDataUrl(filePath: string): Promise<string> {
  const buf = fs.readFileSync(filePath);
  return `data:image/png;base64,${buf.toString('base64')}`;
}

async function dataUrlToGrayStats(dataUrl: string): Promise<{
  width: number;
  height: number;
  intermediateCount: number;
  totalPixels: number;
}> {
  const commaIdx = dataUrl.indexOf(',');
  const b64 = commaIdx >= 0 ? dataUrl.slice(commaIdx + 1) : dataUrl;
  const buf = Buffer.from(b64, 'base64');
  const { data, info } = await sharp(buf).grayscale().raw().toBuffer({ resolveWithObject: true });
  let intermediateCount = 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v >= 15 && v <= 240) intermediateCount++;
  }
  return { width: info.width, height: info.height, intermediateCount, totalPixels: data.length };
}

function makeAntiAliasedStrokeRgba(w: number, h: number): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      // Draw smooth diagonal stroke with anti-aliased edge ramp 25..220 on 250 paper
      const dist = Math.abs((y - 20) - 0.35 * (x - 20));
      let gray = 250;
      if (dist <= 1.2) gray = 25;
      else if (dist <= 3.5) gray = Math.round(25 + ((dist - 1.2) / 2.3) * 200);
      rgba[idx] = gray;
      rgba[idx + 1] = gray;
      rgba[idx + 2] = gray;
      rgba[idx + 3] = 255;
    }
  }
  return rgba;
}

function makeSlantedStemsGray(w: number, h: number, angleDeg: number): Uint8ClampedArray {
  const gray = new Uint8ClampedArray(w * h).fill(255);
  const cy = (h - 1) * 0.5;
  const tanA = Math.tan((angleDeg * Math.PI) / 180);
  const stemPositions = [50, 90, 130, 170];
  for (const cx of stemPositions) {
    for (let y = 12; y < h - 12; y++) {
      // Forward +angleDeg leans top (y < cy) to the right (+x) and bottom (y > cy) to the left (-x)
      const xCenter = cx - (y - cy) * tanA;
      for (let dx = -2; dx <= 2; dx++) {
        const px = Math.round(xCenter + dx);
        if (px >= 0 && px < w) {
          const dist = Math.abs(xCenter + dx - px) + Math.abs(dx) * 0.5;
          const val = Math.min(255, Math.round(20 + dist * 40));
          const idx = y * w + px;
          if (val < gray[idx]) gray[idx] = val;
        }
      }
    }
  }
  return gray;
}

function columnProjectionVariance(gray: Uint8ClampedArray, w: number, h: number): number {
  const cols = new Float64Array(w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (gray[y * w + x] < 160) cols[x]++;
    }
  }
  let sum = 0;
  for (let x = 0; x < w; x++) sum += cols[x];
  const mean = sum / w;
  let varSum = 0;
  for (let x = 0; x < w; x++) {
    const d = cols[x] - mean;
    varSum += d * d;
  }
  return varSum / w;
}

function countInkPixels(gray: Uint8ClampedArray, threshold = 160): number {
  let c = 0;
  for (let i = 0; i < gray.length; i++) {
    if (gray[i] < threshold) c++;
  }
  return c;
}

function cleanupExportDir(ctx: TestContext, exportDir?: string): void {
  if (!exportDir) return;
  ctx.registerCleanup(() => {
    try {
      fs.rmSync(exportDir, { recursive: true, force: true });
      const parent = path.dirname(exportDir);
      if (fs.existsSync(parent) && fs.readdirSync(parent).length === 0) {
        fs.rmdirSync(parent);
      }
    } catch {
      // ignore
    }
  });
}

export const tier1Tests: TestCase[] = [
  // =========================================================================
  // F1: Grayscale-Preserving Normalization (R1) — 6 Tests
  // =========================================================================
  {
    id: 'T1-F1-01',
    tier: 1,
    feature: 'F1',
    name: 'Anti-aliased grayscale stroke preservation in preprocessForOCR',
    async run() {
      const w = 160;
      const h = 80;
      const rgba = makeAntiAliasedStrokeRgba(w, h);
      const out = preprocessForOCR({ data: rgba, width: w, height: h });
      assertEqual(out.width, w, 'Output width');
      assertEqual(out.height, h, 'Output height');

      let intermediate = 0;
      let darkInk = 0;
      for (let i = 0; i < w * h; i++) {
        const idx = i * 4;
        const r = out.data[idx];
        assertEqual(out.data[idx + 1], r, 'G === R');
        assertEqual(out.data[idx + 2], r, 'B === R');
        assertEqual(out.data[idx + 3], 255, 'Alpha === 255');
        if (r < 120) darkInk++;
        if (r >= 10 && r <= 240) intermediate++;
      }
      assertGreaterOrEqual(darkInk, 40, 'Should preserve dark stroke pixels');
      assertGreaterOrEqual(intermediate, 30, 'Must preserve anti-aliased intermediate gray pixels (10..240)');
    },
  },
  {
    id: 'T1-F1-02',
    tier: 1,
    feature: 'F1',
    name: 'Uneven shadow lighting normalization preserves stroke gray gradient',
    async run() {
      const w = 180;
      const h = 90;
      const rgba = new Uint8ClampedArray(w * h * 4);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const bg = Math.round(245 - (x / w) * 70 - (y / h) * 70);
          const dist = Math.abs(y - 45);
          let gray = bg;
          if (dist <= 1.2) gray = Math.max(10, bg - 150);
          else if (dist <= 3.2) gray = Math.max(20, Math.round(bg - 150 * (1 - (dist - 1.2) / 2.2)));
          const idx = (y * w + x) * 4;
          rgba[idx] = gray;
          rgba[idx + 1] = gray;
          rgba[idx + 2] = gray;
          rgba[idx + 3] = 255;
        }
      }
      const out = preprocessForOCR({ data: rgba, width: w, height: h });
      // Sample background corners (top-left vs bottom-right, away from y=45 stroke)
      const topLeftBg = out.data[(5 * w + 5) * 4];
      const botRightBg = out.data[((h - 6) * w + (w - 6)) * 4];
      assertLessOrEqual(Math.abs(topLeftBg - botRightBg), 50, 'Background shadow should be flattened');
      let strokeDark = 0;
      let antiAliased = 0;
      for (let x = 10; x < w - 10; x++) {
        const core = out.data[(45 * w + x) * 4];
        const edge = out.data[(47 * w + x) * 4];
        if (core < 110) strokeDark++;
        if (edge >= 20 && edge <= 235) antiAliased++;
      }
      assertGreaterOrEqual(strokeDark, w - 30, 'Stroke center stays dark');
      assertGreaterOrEqual(antiAliased, 20, 'Anti-aliased edge gradients preserved');
    },
  },
  {
    id: 'T1-F1-03',
    tier: 1,
    feature: 'F1',
    name: 'Alpha-composited transparent PNG stroke preservation',
    async run() {
      const w = 120;
      const h = 60;
      const rgba = new Uint8ClampedArray(w * h * 4);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const idx = (y * w + x) * 4;
          const dist = Math.abs(y - 30);
          if (dist <= 1.2) {
            rgba[idx] = 0;
            rgba[idx + 1] = 0;
            rgba[idx + 2] = 0;
            rgba[idx + 3] = 255;
          } else if (dist <= 3.0) {
            rgba[idx] = 20;
            rgba[idx + 1] = 20;
            rgba[idx + 2] = 20;
            rgba[idx + 3] = 128; // semi-transparent edge
          } else {
            rgba[idx] = 0;
            rgba[idx + 1] = 0;
            rgba[idx + 2] = 0;
            rgba[idx + 3] = 0; // fully transparent background
          }
        }
      }
      const out = preprocessForOCR({ data: rgba, width: w, height: h });
      const bgPixel = out.data[(5 * w + 5) * 4];
      assertGreaterOrEqual(bgPixel, 235, 'Transparent background must composite to white (~255), not black');
      const edgePixel = out.data[(32 * w + 50) * 4];
      assert(edgePixel >= 20 && edgePixel <= 245, `Semi-transparent edge expected in [20..245], got ${edgePixel}`);
    },
  },
  {
    id: 'T1-F1-04',
    tier: 1,
    feature: 'F1',
    name: 'Client OcrStudio upload preserves grayscale payload to API in browser (Offline Local + Vision AI)',
    timeoutMs: 200_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();

      // 1. Offline Local upload -> /api/local-ocr preserves grayscale
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.enableNetworkCapture();
      await dom.clickModeButton(/Offline Local/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'cursive-clean-2lines.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);

      const localReqs = dom.getCapturedRequests().filter((r) => r.url.includes('/api/local-ocr') && r.method === 'POST');
      assertGreaterOrEqual(localReqs.length, 1, 'Expected POST /api/local-ocr request');
      const localImg = localReqs[0].parsedBody?.image;
      assert(typeof localImg === 'string' && localImg.startsWith('data:image/'), 'Expected base64 image data URL');
      const localStats = await dataUrlToGrayStats(localImg);
      assertGreaterOrEqual(
        localStats.intermediateCount,
        40,
        'Offline Local uploaded payload must preserve anti-aliased intermediate gray pixels (15..240)'
      );

      // 2. Vision AI / Auto mode -> normalizeImageForOcr -> preprocessForOCR -> POST /api/extract-reminder (R1)
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.enableNetworkCapture();
      if (await dom.hasModeButton(/Vision AI/i)) {
        await dom.clickModeButton(/Vision AI/i);
      } else {
        await dom.clickModeButton(/^Auto$/i);
      }
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'cursive-clean-2lines.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);

      const visionReqs = dom
        .getCapturedRequests()
        .filter(
          (r) =>
            r.url.includes('/api/extract-reminder') &&
            r.method === 'POST' &&
            typeof r.parsedBody?.image === 'string'
        );
      assertGreaterOrEqual(
        visionReqs.length,
        1,
        'Expected POST /api/extract-reminder with client-preprocessed image payload'
      );
      const visionImg = String(visionReqs[0].parsedBody?.image);
      assert(visionImg.startsWith('data:image/'), 'Expected base64 image data URL in /api/extract-reminder');
      const visionStats = await dataUrlToGrayStats(visionImg);
      assertGreaterOrEqual(
        visionStats.intermediateCount,
        40,
        'Vision AI / Auto uploaded payload must preserve anti-aliased intermediate gray pixels (15..240)'
      );
    },
  },
  {
    id: 'T1-F1-05',
    tier: 1,
    feature: 'F1',
    name: 'preprocessImageData separates smooth enhancedGray from strict binary mask',
    async run(ctx) {
      const img = await pngToRgbaImageData(path.join(ctx.fixturesDir, 'cursive-clean-2lines.png'));
      const res = preprocessImageData(img, {
        profile: 'handwriting',
        binarize: 'sauvola',
        flattenLighting: true,
        enhanceContrast: true,
      });
      assert(res.enhancedGray && res.enhancedGray.length === img.width * img.height, 'enhancedGray length mismatch');
      let intermediateGray = 0;
      for (let i = 0; i < res.binary.length; i++) {
        assert(res.binary[i] === 0 || res.binary[i] === 255, 'binary must only contain 0 or 255');
        const g = res.enhancedGray[i];
        if (g > 15 && g < 240) intermediateGray++;
      }
      assertGreaterOrEqual(intermediateGray, 50, 'enhancedGray must retain anti-aliased edge values');
    },
  },
  {
    id: 'T1-F1-06',
    tier: 1,
    feature: 'F1',
    name: 'Faint pencil stroke survival in grayscale normalization',
    async run() {
      const w = 140;
      const h = 60;
      const rgba = new Uint8ClampedArray(w * h * 4);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const isStroke = Math.abs(y - 30) <= 1 && x >= 20 && x <= 120;
          const v = isStroke ? 135 : 245;
          const idx = (y * w + x) * 4;
          rgba[idx] = v;
          rgba[idx + 1] = v;
          rgba[idx + 2] = v;
          rgba[idx + 3] = 255;
        }
      }
      const out = preprocessForOCR({ data: rgba, width: w, height: h });
      const bgSample = out.data[(5 * w + 5) * 4];
      let strokeSum = 0;
      let strokeCount = 0;
      for (let x = 25; x <= 115; x++) {
        strokeSum += out.data[(30 * w + x) * 4];
        strokeCount++;
      }
      const strokeMean = strokeSum / strokeCount;
      assertGreaterOrEqual(bgSample - strokeMean, 40, 'Faint pencil stroke must remain clearly darker than paper');
    },
  },

  // =========================================================================
  // F2: Cursive Slant Normalization (R2) — 6 Tests
  // =========================================================================
  {
    id: 'T1-F2-01',
    tier: 1,
    feature: 'F2',
    name: 'Positive forward slant (+22 deg) estimation and shear normalization',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/geometry.ts');
      const estFn = resolveExportFn<(b: Uint8ClampedArray, w: number, h: number, opts?: Record<string, unknown>) => number>(
        mod,
        ['estimateSlantAngle', 'estimateCursiveSlantAngle']
      );
      const deslantFn = resolveExportFn<(...args: any[]) => Record<string, any>>(mod, ['deslantGrayscale']);
      assert(estFn && deslantFn, 'geometry.ts slant exports missing');

      const w = 240;
      const h = 80;
      const gray = makeSlantedStemsGray(w, h, 22);
      const binary = binarizeSauvola(gray, w, h, 15, 0.2);
      const angle = estFn(gray, w, h, { minAngleDeg: -38, maxAngleDeg: 38 });
      assert(Math.abs(angle - 22) <= 4, `Expected estimated angle near +22°, got ${angle}`);

      const res = deslantFn(gray, binary, w, h, { minAngleDeg: -38, maxAngleDeg: 38 });
      const applied = Boolean(res.applied ?? res.corrected ?? res.deslanted);
      assertEqual(applied, true, 'Slant correction should be applied');
      assertGreaterOrEqual(res.width, w, 'Output width must not shrink');
      const varBefore = columnProjectionVariance(gray, w, h);
      const varAfter = columnProjectionVariance(res.data, res.width, res.height);
      assertGreaterOrEqual(varAfter, varBefore * 1.35, 'Vertical column variance should increase >= 1.35x');
    },
  },
  {
    id: 'T1-F2-02',
    tier: 1,
    feature: 'F2',
    name: 'Negative backward slant (-18 deg) estimation and shear normalization',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/geometry.ts');
      const estFn = resolveExportFn<(b: Uint8ClampedArray, w: number, h: number, opts?: Record<string, unknown>) => number>(
        mod,
        ['estimateSlantAngle', 'estimateCursiveSlantAngle']
      );
      const deslantFn = resolveExportFn<(...args: any[]) => Record<string, any>>(mod, ['deslantGrayscale']);
      assert(estFn && deslantFn, 'geometry.ts slant exports missing');

      const w = 240;
      const h = 80;
      const gray = makeSlantedStemsGray(w, h, -18);
      const binary = binarizeSauvola(gray, w, h, 15, 0.2);
      const angle = estFn(gray, w, h, { minAngleDeg: -38, maxAngleDeg: 38 });
      assert(Math.abs(angle - -18) <= 4, `Expected estimated angle near -18°, got ${angle}`);

      const res = deslantFn(gray, binary, w, h, { minAngleDeg: -38, maxAngleDeg: 38 });
      assert(Boolean(res.applied ?? res.corrected ?? res.deslanted), 'Expected deslant applied');
      const varBefore = columnProjectionVariance(gray, w, h);
      const varAfter = columnProjectionVariance(res.data, res.width, res.height);
      assertGreaterOrEqual(varAfter, varBefore * 1.3, 'Vertical column variance should increase >= 1.3x');
    },
  },
  {
    id: 'T1-F2-03',
    tier: 1,
    feature: 'F2',
    name: 'Zero character clipping and ink conservation on edge strokes',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/geometry.ts');
      const shearFn = resolveExportFn<
        (g: Uint8ClampedArray, w: number, h: number, deg: number, fill?: number) => {
          data: Uint8ClampedArray;
          width: number;
          height: number;
        }
      >(mod, ['shearDeslantGrayscale', 'shearGrayscale']);
      assert(shearFn, 'shearDeslantGrayscale / shearGrayscale export missing');

      const w = 140;
      const h = 70;
      const gray = new Uint8ClampedArray(w * h).fill(255);
      const cy = (h - 1) * 0.5;
      const tan24 = Math.tan((24 * Math.PI) / 180);
      // Draw stems near left edge (x=18) and right edge (x=w-19)
      for (const cx of [18, w - 19]) {
        for (let y = 6; y < h - 6; y++) {
          const px = Math.round(cx - (y - cy) * tan24);
          if (px >= 1 && px < w - 1) {
            gray[y * w + px] = 15;
            gray[y * w + px + 1] = 15;
          }
        }
      }
      const inkBefore = countInkPixels(gray);
      const out = shearFn(gray, w, h, 24, 255);
      assertGreaterOrEqual(out.width, w + Math.ceil(h * Math.abs(Math.tan((24 * Math.PI) / 180))), 'Expanded width');
      assertEqual(out.height, h, 'Height preserved');
      const inkAfter = countInkPixels(out.data);
      const ratio = inkAfter / Math.max(1, inkBefore);
      assert(ratio >= 0.9 && ratio <= 1.15, `Ink conservation ratio out of bounds: ${ratio}`);
      // Outer left & right border columns remain paper white
      for (let y = 0; y < out.height; y++) {
        assertEqual(out.data[y * out.width], 255, 'Leftmost border column must remain 255');
        assertEqual(out.data[y * out.width + out.width - 1], 255, 'Rightmost border column must remain 255');
      }
    },
  },
  {
    id: 'T1-F2-04',
    tier: 1,
    feature: 'F2',
    name: 'Upright 0-degree text bypasses unnecessary shear',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/geometry.ts');
      const deslantFn = resolveExportFn<(...args: any[]) => Record<string, any>>(mod, ['deslantGrayscale']);
      assert(deslantFn, 'deslantGrayscale missing');

      const w = 160;
      const h = 60;
      const gray = makeSlantedStemsGray(w, h, 0);
      const binary = binarizeSauvola(gray, w, h, 15, 0.2);
      const res = deslantFn(gray, binary, w, h);
      const applied = Boolean(res.applied ?? res.corrected ?? res.deslanted);
      assertEqual(applied, false, 'Upright text must not trigger deslanting');
      assertEqual(res.angleDeg, 0, 'angleDeg should be 0');
      assertEqual(res.width, w, 'Width unchanged');
    },
  },
  {
    id: 'T1-F2-05',
    tier: 1,
    feature: 'F2',
    name: 'Direct shearDeslantGrayscale API contract & white corner fill',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/geometry.ts');
      const shearFn = resolveExportFn<
        (g: Uint8ClampedArray, w: number, h: number, deg: number, fill?: number) => {
          data: Uint8ClampedArray;
          width: number;
          height: number;
        }
      >(mod, ['shearDeslantGrayscale', 'shearGrayscale']);
      assert(shearFn, 'shearDeslantGrayscale / shearGrayscale missing');
      const w = 100;
      const h = 50;
      const gray = new Uint8ClampedArray(w * h).fill(220);
      for (let y = 10; y < 40; y++) gray[y * w + 50] = 30;
      const out = shearFn(gray, w, h, 15, 255);
      assertEqual(out.height, h, 'Height unchanged');
      assertGreaterOrEqual(out.width, w + 4, 'Width expanded for +15 deg shear');
      assertEqual(out.data[0], 255, 'Top-left exposed corner filled with 255');
      assertEqual(out.data[out.width - 1], 255, 'Top-right exposed corner filled with 255');
      for (let i = 0; i < out.data.length; i++) {
        assert(!Number.isNaN(out.data[i]) && out.data[i] >= 0 && out.data[i] <= 255, 'Valid pixel byte');
      }
    },
  },
  {
    id: 'T1-F2-06',
    tier: 1,
    feature: 'F2',
    name: 'Multi-word slanted cursive PNG fixture slant normalization',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/geometry.ts');
      const deslantFn = resolveExportFn<(...args: any[]) => Record<string, any>>(mod, ['deslantGrayscale']);
      assert(deslantFn, 'deslantGrayscale missing');

      const { data: gray, width, height } = await pngToGrayBuffer(
        path.join(ctx.fixturesDir, 'cursive-slanted-pos22.png')
      );
      const binary = binarizeSauvola(gray, width, height, 15, 0.2);
      const res = deslantFn(gray, binary, width, height, { minAngleDeg: -38, maxAngleDeg: 38 });
      assert(Boolean(res.applied ?? res.corrected ?? res.deslanted), 'Expected slant correction on +22 deg PNG');
      assertGreaterOrEqual(Math.abs(res.angleDeg), 10, 'Detected slant angle should be >= 10 deg');
      const inkBefore = countInkPixels(gray, 180);
      const inkAfter = countInkPixels(res.data, 180);
      assertGreaterOrEqual(inkAfter, inkBefore * 0.85, 'Edge characters must not be clipped');
    },
  },

  // =========================================================================
  // F3: Ascender/Descender Line Segmentation & Cleanup (R2, Finding 1) — 6 Tests
  // =========================================================================
  {
    id: 'T1-F3-01',
    tier: 1,
    feature: 'F3',
    name: 'Multi-line ascender/descender (f,g,j,l,p,q,y) preservation',
    async run(ctx) {
      const { data: gray, width, height } = await pngToGrayBuffer(
        path.join(ctx.fixturesDir, 'cursive-tight-descenders.png')
      );
      const binary = binarizeSauvola(gray, width, height, 19, 0.2);
      const boxes = segmentLines(binary, width, height);
      assertEqual(boxes.length, 3, 'Must detect all 3 tightly spaced lines');
      assert(boxes[0].y0 < boxes[1].y0 && boxes[1].y0 < boxes[2].y0, 'Boxes must be strictly ordered top-to-bottom');

      for (const box of boxes) {
        const cropped = cropLine(gray, width, height, box, 14);
        assertGreaterOrEqual(cropped.height, 40, 'Cropped line height should preserve full glyph vertical span');
        // Check that top/bottom white pad exists around strokes (not sheared through glyph top/bottom)
        let topRowInk = 0;
        let botRowInk = 0;
        for (let x = 0; x < cropped.width; x++) {
          if (cropped.data[x] < 180) topRowInk++;
          if (cropped.data[(cropped.height - 1) * cropped.width + x] < 180) botRowInk++;
        }
        assertEqual(topRowInk, 0, 'Top pad row must be clean paper (no clipped ascender)');
        assertEqual(botRowInk, 0, 'Bottom pad row must be clean paper (no clipped descender)');
      }
    },
  },
  {
    id: 'T1-F3-02',
    tier: 1,
    feature: 'F3',
    name: 'cropLine extracts real grayscale context before adding white padding',
    async run() {
      const w = 120;
      const h = 80;
      const gray = new Uint8ClampedArray(w * h).fill(255);
      // Core line in y=25..40, faint descender tail extending to y=44
      for (let y = 25; y <= 40; y++) {
        for (let x = 20; x <= 90; x++) gray[y * w + x] = 35;
      }
      for (let y = 41; y <= 44; y++) {
        gray[y * w + 55] = 90; // faint descender outside tight box
      }
      const cropped = cropLine(gray, w, h, { x0: 20, y0: 25, x1: 90, y1: 40 }, 16);
      // Outer top row (pad=16) is white 255
      assertEqual(cropped.data[0], 255, 'Outer white padding must be 255');
      // Verify faint descender (4px below y1=40) is captured from real gray context inside padded region!
      let foundDescender = false;
      for (let y = 0; y < cropped.height; y++) {
        for (let x = 0; x < cropped.width; x++) {
          if (cropped.data[y * cropped.width + x] === 90) foundDescender = true;
        }
      }
      assert(foundDescender, 'cropLine must extract real grayscale pixels around box before white fill');
    },
  },
  {
    id: 'T1-F3-03',
    tier: 1,
    feature: 'F3',
    name: 'Ligature-safe splitWideLineBox splits only at inter-word gaps',
    async run() {
      const w = 740;
      const h = 70;
      const binary = new Uint8ClampedArray(w * h).fill(255);
      // 3 dense connected words (each 200px < maxChunkW=253) separated by 40px whitespace gaps
      const wordSpans: Array<[number, number]> = [
        [20, 220],
        [260, 460],
        [500, 700],
      ];
      for (const [x0, x1] of wordSpans) {
        for (let y = 20; y <= 50; y++) {
          for (let x = x0; x <= x1; x++) binary[y * w + x] = 0;
        }
      }
      const chunks = splitWideLineBox(binary, w, h, { x0: 20, y0: 16, x1: 700, y1: 54 }, 6.5);
      assertGreaterOrEqual(chunks.length, 2, 'Wide 3-word line should split into multiple chunks');
      for (const chunk of chunks) {
        for (const [wx0, wx1] of wordSpans) {
          // Chunk boundary must NEVER cut strictly inside a dense word span
          const cutsStart = chunk.x0 > wx0 + 2 && chunk.x0 < wx1 - 2;
          const cutsEnd = chunk.x1 > wx0 + 2 && chunk.x1 < wx1 - 2;
          assert(!cutsStart && !cutsEnd, `Chunk [${chunk.x0}..${chunk.x1}] bisected dense word [${wx0}..${wx1}]`);
        }
      }
    },
  },
  {
    id: 'T1-F3-04',
    tier: 1,
    feature: 'F3',
    name: 'Continuous cursive word with ligature necks avoids dense mid-stroke slice',
    async run() {
      const w = 480;
      const h = 70;
      const binary = new Uint8ClampedArray(w * h).fill(255);
      // Continuous connected cursive body from x=10..470 (boxH=45 => maxChunkW=292, searchWindow=[160..292])
      // with one thin 2px-tall ligature neck at x=240..255 inside [160..292]
      for (let x = 10; x <= 470; x++) {
        const isNeck = x >= 240 && x <= 255;
        const yMin = isNeck ? 34 : 14;
        const yMax = isNeck ? 35 : 58;
        for (let y = yMin; y <= yMax; y++) {
          binary[y * w + x] = 0;
        }
      }
      const chunks = splitWideLineBox(binary, w, h, { x0: 10, y0: 14, x1: 470, y1: 58 }, 6.5);
      assertGreaterOrEqual(
        chunks.length,
        2,
        'Continuous wide cursive word must split into >= 2 chunks at ligature neck'
      );
      const cutX = chunks[0].x1;
      assert(
        cutX >= 238 && cutX <= 257,
        `Continuous word must split at minimum-ink ligature neck (238..257), got cutX=${cutX}`
      );
    },
  },
  {
    id: 'T1-F3-05',
    tier: 1,
    feature: 'F3',
    name: 'Left-border spiral margin coil rejection (Finding 1)',
    async run(ctx) {
      const { data: gray, width, height } = await pngToGrayBuffer(
        path.join(ctx.fixturesDir, 'cursive-spiral-bleed.png')
      );
      const binary = binarizeSauvola(gray, width, height, 19, 0.2);
      const boxes = segmentLines(binary, width, height);
      assertEqual(boxes.length, 3, 'Should detect strictly the 3 interior text lines (ignoring left spiral holes)');
      for (const b of boxes) {
        assertGreaterOrEqual(b.x0, 40, `Line box x0=${b.x0} leaked into left spiral margin`);
      }
    },
  },
  {
    id: 'T1-F3-06',
    tier: 1,
    feature: 'F3',
    name: 'Top-right corner edge bleed rejection (Finding 1)',
    async run() {
      const w = 400;
      const h = 240;
      const binary = new Uint8ClampedArray(w * h).fill(255);
      // Dark corner fold/bleed at top-right (w-26..w-1, 0..28)
      for (let y = 0; y < 28; y++) {
        for (let x = w - 26 + y; x < w; x++) {
          if (x >= 0 && x < w) binary[y * w + x] = 0;
        }
      }
      // Single valid centered text line at y=110..138, x=80..260
      for (let y = 112; y <= 136; y++) {
        for (let x = 80; x <= 260; x++) {
          if (x % 9 < 5) binary[y * w + x] = 0;
        }
      }
      const cleaned = filterTextComponents(binary, w, h);
      const boxes = segmentLines(cleaned, w, h);
      assertEqual(boxes.length, 1, 'Only the centered valid line should remain after corner bleed cleanup');
      assertGreaterOrEqual(boxes[0].y0, 80, 'Top-right corner bleed must not produce top line box');
      assertLessOrEqual(boxes[0].x1, 300, 'Line box must not span to top-right bleed');
    },
  },

  // =========================================================================
  // F4: TrOCR Beam Search, Padding & Ensemble (R3, Findings 2 & 4) — 6 Tests
  // =========================================================================
  {
    id: 'T1-F4-01',
    tier: 1,
    feature: 'F4',
    name: 'TrOCR handwriting beam-search recognition of clean 2-line cursive note',
    timeoutMs: 75_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const fp = path.join(ctx.fixturesDir, 'cursive-clean-2lines.png');
      const gt = ctx.manifest.fixtures['cursive-clean-2lines.png'].groundTruth;
      const image = await pngToDataUrl(fp);
      const { status, body } = await srv.postLocalOcr({ image, ocrProfile: 'handwriting', exportStages: false }, 55_000);
      assertEqual(status, 200, 'HTTP 200 expected');
      assertEqual(body.success, true, 'success should be true');
      assertEqual(body.recognizer, 'trocr-host', 'Must use trocr-host for handwriting profile');
      assertEqual(body.lines.length, 2, 'Expected 2 segmented lines');
      const acc = characterAccuracy(body.text, gt, { caseInsensitive: true });
      assertGreaterOrEqual(acc, 0.85, `Expected 1-CER >= 0.85 on clean 2-line cursive, got ${acc.toFixed(3)} ("${body.text}")`);
    },
  },
  {
    id: 'T1-F4-02',
    tier: 1,
    feature: 'F4',
    name: 'No hardcoded canonicalGroceryItems mock fallback on >=6 lines',
    timeoutMs: 75_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const fp = path.join(ctx.fixturesDir, 'non-grocery-7lines.png');
      const image = await pngToDataUrl(fp);
      const { status, body } = await srv.postLocalOcr({ image, ocrProfile: 'handwriting', exportStages: false }, 55_000);
      assertEqual(status, 200, 'HTTP 200 expected');
      const lower = body.text.toLowerCase();
      assertNotIncludes(lower, 'pan au chocolat', 'Must not emit hardcoded grocery mock');
      assertNotIncludes(lower, 'turkey mince', 'Must not emit hardcoded grocery mock');
      assertNotIncludes(lower, 'sundried tomatoes', 'Must not emit hardcoded grocery mock');
      const gt = ctx.manifest.fixtures['non-grocery-7lines.png'].groundTruth;
      const acc = characterAccuracy(body.text, gt, { caseInsensitive: true });
      assertGreaterOrEqual(acc, 0.85, `Expected genuine TrOCR accuracy >= 0.85 on 7 non-grocery lines, got ${acc.toFixed(3)} ("${body.text}")`);
    },
  },
  {
    id: 'T1-F4-03',
    tier: 1,
    feature: 'F4',
    name: 'Long line decoding beyond 48-token cap (max_new_tokens: 64)',
    timeoutMs: 75_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const fp = path.join(ctx.fixturesDir, 'cursive-long-line.png');
      const gt = ctx.manifest.fixtures['cursive-long-line.png'].groundTruth;
      const image = await pngToDataUrl(fp);
      const { status, body } = await srv.postLocalOcr({ image, ocrProfile: 'handwriting', exportStages: false }, 55_000);
      assertEqual(status, 200, 'HTTP 200 expected');
      const lower = body.text.toLowerCase();
      assertIncludes(lower, 'remember', 'Must capture start of long line');
      assert(
        lower.includes('nine') || lower.includes('laptop'),
        `Long line truncated before end: "${body.text}"`
      );
      const acc = characterAccuracy(body.text, gt, { caseInsensitive: true });
      assertGreaterOrEqual(acc, 0.85, `Expected >= 0.85 accuracy on long line, got ${acc.toFixed(3)}`);
    },
  },
  {
    id: 'T1-F4-04',
    tier: 1,
    feature: 'F4',
    name: 'TrOCR vertical crop padding >= 14px (Finding 4)',
    timeoutMs: 120_000,
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'server/handwritingRecognizer.ts');
      const prepFn = resolveExportFn<
        (crop: { data: Uint8ClampedArray; width: number; height: number }) => Promise<{
          data: Uint8Array | Uint8ClampedArray;
          width: number;
          height: number;
          channels?: number;
        }>
      >(mod, ['prepareTrOcrRawImage']);

      const measureVerticalPadding = (
        buf: Uint8Array | Uint8ClampedArray,
        width: number,
        height: number,
        channels: number
      ) => {
        let topPadRows = 0;
        for (let y = 0; y < height; y++) {
          let hasInk = false;
          for (let x = 0; x < width; x++) {
            const base = (y * width + x) * channels;
            for (let c = 0; c < channels; c++) {
              if (buf[base + c] < 245) {
                hasInk = true;
                break;
              }
            }
            if (hasInk) break;
          }
          if (hasInk) break;
          topPadRows++;
        }
        let botPadRows = 0;
        for (let y = height - 1; y >= 0; y--) {
          let hasInk = false;
          for (let x = 0; x < width; x++) {
            const base = (y * width + x) * channels;
            for (let c = 0; c < channels; c++) {
              if (buf[base + c] < 245) {
                hasInk = true;
                break;
              }
            }
            if (hasInk) break;
          }
          if (hasInk) break;
          botPadRows++;
        }
        return { topPadRows, botPadRows };
      };

      // (a) Direct contract check on prepareTrOcrRawImage if exported
      if (prepFn) {
        const cw = 120;
        const ch = 32;
        const tightCrop = new Uint8ClampedArray(cw * ch).fill(255);
        // Draw dark strokes touching top (y=0) and bottom (y=ch-1)
        for (let y = 0; y < ch; y++) {
          for (let x = 16; x < cw - 16; x += 12) {
            tightCrop[y * cw + x] = 20;
            tightCrop[y * cw + x + 1] = 20;
          }
        }
        const rawImg = await prepFn({ data: tightCrop, width: cw, height: ch });
        const channels = rawImg.channels ?? Math.max(1, Math.round(rawImg.data.length / (rawImg.width * rawImg.height)));
        const { topPadRows, botPadRows } = measureVerticalPadding(rawImg.data, rawImg.width, rawImg.height, channels);
        assertGreaterOrEqual(topPadRows, 14, `prepareTrOcrRawImage top white padding must be >= 14px, got ${topPadRows}`);
        assertGreaterOrEqual(botPadRows, 14, `prepareTrOcrRawImage bottom white padding must be >= 14px, got ${botPadRows}`);
      }

      // (b) Optional 06b_trocr_input_*.png stage export verification with isolated exportRootDir
      const procFn = resolveExportFn<(opts: Record<string, unknown>) => Promise<Record<string, any>>>(
        mod,
        ['processImageWithLocalOcr']
      );
      assert(procFn, 'processImageWithLocalOcr not found');
      const fp = path.join(ctx.fixturesDir, 'cursive-tight-descenders.png');
      const gt = ctx.manifest.fixtures['cursive-tight-descenders.png'].groundTruth;
      const exportRootDir = ctx.createTempDir('ocr-exports-');
      const stageRes = await procFn({
        imageBuffer: fs.readFileSync(fp),
        ocrProfile: 'handwriting',
        exportStages: true,
        exportRootDir,
      });
      cleanupExportDir(ctx, stageRes.exportDir);
      if (stageRes.exportDir && fs.existsSync(stageRes.exportDir)) {
        const trocrStagePngs = fs
          .readdirSync(stageRes.exportDir)
          .filter((f) => /^06b_trocr_input_.*\.png$/i.test(f));
        for (const file of trocrStagePngs) {
          const { data, width, height } = await pngToGrayBuffer(path.join(stageRes.exportDir, file));
          const { topPadRows, botPadRows } = measureVerticalPadding(data, width, height, 1);
          assertGreaterOrEqual(topPadRows, 14, `${file} top vertical padding must be >= 14px`);
          assertGreaterOrEqual(botPadRows, 14, `${file} bottom vertical padding must be >= 14px`);
        }
      }

      // (c) Unconditional opaque-box POST /api/local-ocr check on tight descenders
      const srv = await ctx.getServer();
      const { status, body } = await srv.postLocalOcr(
        { image: await pngToDataUrl(fp), ocrProfile: 'handwriting', exportStages: false },
        55_000
      );
      assertEqual(status, 200, 'HTTP 200 expected');
      assertEqual(body.lines.length, 3, `Expected 3 detected lines on cursive-tight-descenders.png, got ${body.lines.length}`);
      const acc = characterAccuracy(body.text, gt, { caseInsensitive: true });
      assertGreaterOrEqual(acc, 0.9, `Expected >= 0.90 accuracy on tight descenders, got ${acc.toFixed(3)} ("${body.text}")`);
    },
  },
  {
    id: 'T1-F4-05',
    tier: 1,
    feature: 'F4',
    name: 'Anti-hallucination guard on noisy paper without legible text',
    timeoutMs: 75_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const image = await pngToDataUrl(path.join(ctx.fixturesDir, 'noisy-paper.png'));
      const { status, body } = await srv.postLocalOcr({ image, ocrProfile: 'handwriting', exportStages: false }, 55_000);
      assertEqual(status, 200, 'HTTP 200 expected on noisy paper');
      assertNotIncludes(body.text.toLowerCase(), 'government', 'Must not emit TrOCR repetitive loop hallucination');
      assert(!/0\s+000\s+000/.test(body.text), 'Must not emit repetitive 0 000 000 hallucination');
      assertLessOrEqual(body.text.trim().length, 3, `Noisy paper should yield empty or <= 3 chars, got "${body.text}"`);
    },
  },
  {
    id: 'T1-F4-06',
    tier: 1,
    feature: 'F4',
    name: 'TrOCR ensemble/re-ranking avoids PARSeq false-1.0 cursive misreads (Finding 2)',
    timeoutMs: 75_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const fp = path.join(ctx.fixturesDir, 'cursive-anchors-grocery.png');
      const gt = ctx.manifest.fixtures['cursive-anchors-grocery.png'].groundTruth;
      const image = await pngToDataUrl(fp);
      const { status, body } = await srv.postLocalOcr({ image, ocrProfile: 'handwriting', exportStages: false }, 55_000);
      assertEqual(status, 200, 'HTTP 200 expected');
      const lower = body.text.toLowerCase();
      assertNotIncludes(lower, 'jolmm', 'Must not emit PARSeq salmon misread');
      assertNotIncludes(lower, 'virwlh', 'Must not emit PARSeq broccoli misread');
      assertNotIncludes(lower, 'can an chocolat', 'Must not emit PARSeq chocolat misread');
      const acc = characterAccuracy(body.text, gt, { caseInsensitive: true });
      assertGreaterOrEqual(acc, 0.85, `Expected ensemble accuracy >= 0.85 on grocery cursive words, got ${acc.toFixed(3)} ("${body.text}")`);
    },
  },

  // =========================================================================
  // F5: Contextual Confusion Resolution & Preservation (R3, Finding 3) — 6 Tests
  // =========================================================================
  {
    id: 'T1-F5-01',
    tier: 1,
    feature: 'F5',
    name: 'Bidirectional cursive confusion resolution (rn<->m, cl<->d, vv<->w, ri<->n)',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/contextualPostprocess.ts');
      const fn = resolveExportFn<(text: string, opts?: Record<string, unknown>) => { text: string }>(
        mod,
        ['postprocessOcrText']
      );
      assert(fn, 'src/lib/ocr/contextualPostprocess.ts must export postprocessOcrText');
      const input =
        'buy rnilk\nclate at vvork\nrnorning rneeting\nspinach and suridried\nretum concem pattem\ndlear dass\npnce fnend';
      const out = fn(input, { style: 'handwriting' });
      const lower = out.text.toLowerCase();
      assertIncludes(lower, 'milk', 'rnilk -> milk (rn -> m)');
      assertIncludes(lower, 'morning', 'rnorning -> morning (rn -> m)');
      assertIncludes(lower, 'meeting', 'rneeting -> meeting (rn -> m)');
      assertIncludes(lower, 'return', 'retum -> return (m -> rn)');
      assertIncludes(lower, 'concern', 'concem -> concern (m -> rn)');
      assertIncludes(lower, 'pattern', 'pattem -> pattern (m -> rn)');
      assertIncludes(lower, 'date', 'clate -> date (cl -> d)');
      assertIncludes(lower, 'clear', 'dlear -> clear (d -> cl)');
      assertIncludes(lower, 'class', 'dass -> class (d -> cl)');
      assertIncludes(lower, 'work', 'vvork -> work (vv -> w)');
      assertIncludes(lower, 'sundried', 'suridried -> sundried (ri -> n)');
      assertIncludes(lower, 'price', 'pnce -> price (n -> ri)');
      assertIncludes(lower, 'friend', 'fnend -> friend (n -> ri)');
    },
  },
  {
    id: 'T1-F5-02',
    tier: 1,
    feature: 'F5',
    name: 'Anchor non-overcorrection for tomorrow, tomatoes, turkey (Finding 3)',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/contextualPostprocess.ts');
      const fn = resolveExportFn<(text: string, opts?: Record<string, unknown>) => { text: string }>(
        mod,
        ['postprocessOcrText']
      );
      assert(fn, 'contextualPostprocess.ts -> postprocessOcrText missing');
      const out = fn('lunch for tomorrow\nsundried tomatoes\nturkey mince', { style: 'handwriting' });
      const lower = out.text.toLowerCase();
      assertIncludes(lower, 'tomorrow', 'tomorrow preserved');
      assertIncludes(lower, 'tomatoes', 'tomatoes preserved');
      assertIncludes(lower, 'turkey', 'turkey preserved');
      assertNotIncludes(lower, 'common', 'tomorrow must not become common');
      assertNotIncludes(lower, 'form', 'tomorrow must not become form');
      assertNotIncludes(lower, 'teacher', 'tomatoes must not become teacher');
      assertNotIncludes(lower, 'they', 'turkey must not become they');
    },
  },
  {
    id: 'T1-F5-03',
    tier: 1,
    feature: 'F5',
    name: 'Strict preservation of casing, punctuation, numbers, dates, and currency',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/contextualPostprocess.ts');
      const fn = resolveExportFn<(text: string, opts?: Record<string, unknown>) => { text: string }>(
        mod,
        ['postprocessOcrText']
      );
      assert(fn, 'contextualPostprocess.ts -> postprocessOcrText missing');
      const sample = 'URGENT: Pay Dr. Alice $142.50 by 5/12/2026 at 3:30pm!';
      const out = fn(sample, { style: 'handwriting' });
      assertEqual(out.text, sample, 'Casing, punctuation, numbers, dates, and currency must stay verbatim');
    },
  },
  {
    id: 'T1-F5-04',
    tier: 1,
    feature: 'F5',
    name: 'Preservation of proper nouns and non-lexicon capitalized names',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/contextualPostprocess.ts');
      const fn = resolveExportFn<(text: string, opts?: Record<string, unknown>) => { text: string }>(
        mod,
        ['postprocessOcrText']
      );
      assert(fn, 'contextualPostprocess.ts -> postprocessOcrText missing');
      const sample = 'Meet Sarah Bramwell and Dr. Patel in London';
      const out = fn(sample, { style: 'handwriting' });
      for (const noun of ['Sarah', 'Bramwell', 'Dr.', 'Patel', 'London']) {
        assertIncludes(out.text, noun, `Proper noun ${noun} must remain intact`);
      }
    },
  },
  {
    id: 'T1-F5-05',
    tier: 1,
    feature: 'F5',
    name: 'postprocessOcrLines filters hallucinations and returns accurate line metadata',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/contextualPostprocess.ts');
      const fn = resolveExportFn<
        (lines: Array<{ text: string; confidence: number }>) => {
          lines: Array<{ text: string; confidence: number }>;
          text: string;
          corrections: unknown[];
          accuracyScore: number;
          verifiedWords: number;
          totalWords: number;
        }
      >(mod, ['postprocessOcrLines']);
      assert(fn, 'contextualPostprocess.ts -> postprocessOcrLines missing');
      const out = fn([
        { text: 'buy rnilk', confidence: 0.55 },
        { text: '0 000 000 000 000', confidence: 0.8 },
      ]);
      assertEqual(out.lines.length, 1, 'Hallucination line should be dropped');
      assertIncludes(out.text.toLowerCase(), 'milk', 'Valid line confusion resolved');
      assertGreaterOrEqual(out.totalWords, 2, 'totalWords reported');
      assert(typeof out.accuracyScore === 'number' && !Number.isNaN(out.accuracyScore), 'Numeric accuracyScore');
    },
  },
  {
    id: 'T1-F5-06',
    tier: 1,
    feature: 'F5',
    name: 'Honest unclamped accuracyScore (no 88% floor hack)',
    async run(ctx) {
      const ctxMod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/contextualPostprocess.ts');
      const ctxFn = resolveExportFn<(t: string) => { accuracyScore: number }>(ctxMod, ['postprocessOcrText']);
      assert(ctxFn, 'contextualPostprocess.ts -> postprocessOcrText missing');
      const ctxRes = ctxFn('xyzqwk plmnbv zzzqqq');
      assertLessOrEqual(
        ctxRes.accuracyScore,
        20,
        `contextualPostprocess.ts must not clamp gibberish to 88%; got ${ctxRes.accuracyScore}`
      );

      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/wordDataset.ts');
      const fn = resolveExportFn<(t: string) => { accuracyScore: number }>(mod, ['correctTextWithLexicon']);
      assert(fn, 'wordDataset.ts -> correctTextWithLexicon missing');
      const res = fn('xyzqwk plmnbv zzzqqq');
      assertLessOrEqual(res.accuracyScore, 20, `wordDataset.ts gibberish must not be clamped to 88%; got ${res.accuracyScore}`);
    },
  },

  // =========================================================================
  // F6: OCR Studio UI Modes & Endpoint Routing (R4) — 6 Tests
  // =========================================================================
  {
    id: 'T1-F6-01',
    tier: 1,
    feature: 'F6',
    name: 'UI renders all 4 R4 OCR mode controls (Auto, Cursive/Handwriting, Printed, Offline Local)',
    timeoutMs: 60_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      assert(await dom.hasModeButton(/^Auto$/i), 'Missing Auto mode button');
      assert(
        await dom.hasModeButton(/^Cursive\s*\/\s*Handwriting$/i),
        'Missing Cursive/Handwriting mode button (R4)'
      );
      assert(await dom.hasModeButton(/^Printed$/i), 'Missing exact Printed mode button (R4)');
      assert(await dom.hasModeButton(/^Offline Local$/i), 'Missing Offline Local mode button (R4)');
    },
  },
  {
    id: 'T1-F6-02',
    tier: 1,
    feature: 'F6',
    name: 'Offline Local mode routes strictly to POST /api/local-ocr',
    timeoutMs: 105_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.enableNetworkCapture();
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'cursive-1line.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);

      const reqs = dom.getCapturedRequests();
      assert(
        reqs.some((r) => r.url.includes('/api/local-ocr') && r.method === 'POST'),
        'Offline Local mode must POST to /api/local-ocr'
      );
      assert(
        !reqs.some((r) => r.url.includes('/api/extract-reminder')),
        'Offline Local initial scan must NOT call /api/extract-reminder'
      );
    },
  },
  {
    id: 'T1-F6-03',
    tier: 1,
    feature: 'F6',
    name: 'Cursive/Handwriting mode sends ocrProfile handwriting',
    timeoutMs: 105_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.enableNetworkCapture();
      await dom.clickModeButton(/^Cursive\s*\/\s*Handwriting$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'cursive-1line.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);

      const reqs = dom
        .getCapturedRequests()
        .filter((r) => r.method === 'POST' && (r.url.includes('/api/local-ocr') || r.url.includes('/api/extract-reminder')));
      assertGreaterOrEqual(reqs.length, 1, 'Expected OCR API request');
      assertEqual(reqs[0].parsedBody?.ocrProfile, 'handwriting', 'Must send ocrProfile: handwriting');
    },
  },
  {
    id: 'T1-F6-04',
    tier: 1,
    feature: 'F6',
    name: 'Printed mode sends ocrProfile printed',
    timeoutMs: 105_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.enableNetworkCapture();
      await dom.clickModeButton(/^Printed$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'printed-schedule.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);

      const reqs = dom
        .getCapturedRequests()
        .filter((r) => r.method === 'POST' && (r.url.includes('/api/local-ocr') || r.url.includes('/api/extract-reminder')));
      assertGreaterOrEqual(reqs.length, 1, 'Expected OCR API request');
      assertEqual(reqs[0].parsedBody?.ocrProfile, 'printed', 'Must send ocrProfile: printed');
    },
  },
  {
    id: 'T1-F6-05',
    tier: 1,
    feature: 'F6',
    name: 'Auto mode is default and sends ocrProfile auto',
    timeoutMs: 105_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.enableNetworkCapture();
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'printed-schedule.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);

      const reqs = dom
        .getCapturedRequests()
        .filter((r) => r.method === 'POST' && (r.url.includes('/api/local-ocr') || r.url.includes('/api/extract-reminder')));
      assertGreaterOrEqual(reqs.length, 1, 'Expected OCR API request on default Auto mode');
      assertEqual(reqs[0].parsedBody?.ocrProfile, 'auto', 'Default mode must send ocrProfile: auto');
    },
  },
  {
    id: 'T1-F6-06',
    tier: 1,
    feature: 'F6',
    name: 'Backend /api/local-ocr and /api/extract-reminder honor ocrProfile contracts',
    timeoutMs: 105_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const image = await pngToDataUrl(path.join(ctx.fixturesDir, 'printed-schedule.png'));
      const localRes = await srv.postLocalOcr({ image, ocrProfile: 'printed', exportStages: false }, 45_000);
      assertEqual(localRes.status, 200, '/api/local-ocr status');
      assertEqual(localRes.body.profile, 'printed', '/api/local-ocr profile');

      const visionRes = await srv.postExtractReminder({ image, ocrProfile: 'printed' }, 45_000);
      assertEqual(visionRes.status, 200, '/api/extract-reminder status');
      assertEqual(visionRes.body.ocrProfile, 'printed', '/api/extract-reminder ocrProfile');
    },
  },

  // =========================================================================
  // F7: OCR Studio UI Metadata & Copy Text (R4) — 6 Tests
  // =========================================================================
  {
    id: 'T1-F7-01',
    tier: 1,
    feature: 'F7',
    name: 'UI renders overall Accuracy % badge matching actual scan response',
    timeoutMs: 105_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.enableNetworkCapture();
      await dom.clickModeButton(/^Offline Local$/i);
      const fp = path.join(ctx.fixturesDir, 'printed-schedule.png');
      await dom.uploadFixture(fp);
      await dom.waitForScanComplete(55_000, ctx.signal);

      const badge = await dom.getAccuracyBadge();
      assert(badge.found && badge.percent !== null, `Accuracy % badge missing; got ${JSON.stringify(badge)}`);
      assert(badge.percent > 0 && badge.percent <= 100, `Accuracy % out of range: ${badge.percent}`);

      const reqs = dom.getCapturedRequests().filter((r) => r.url.includes('/api/local-ocr') && r.method === 'POST');
      let rawAcc = reqs[0]?.responseBody?.accuracyScore as number | undefined;
      if (typeof rawAcc !== 'number') {
        const direct = await srv.postLocalOcr(
          { image: await pngToDataUrl(fp), ocrProfile: 'auto', exportStages: false },
          45_000
        );
        rawAcc = direct.body.accuracyScore;
      }
      assert(typeof rawAcc === 'number', `Expected numeric accuracyScore from /api/local-ocr, got ${rawAcc}`);
      const expectedAccuracyPct = rawAcc <= 1 && rawAcc > 0 ? Math.round(rawAcc * 100) : Math.round(rawAcc);
      assertLessOrEqual(
        Math.abs(badge.percent - expectedAccuracyPct),
        1,
        `UI Accuracy % badge (${badge.percent}%) must correlate with scan response (${expectedAccuracyPct}%)`
      );
    },
  },
  {
    id: 'T1-F7-02',
    tier: 1,
    feature: 'F7',
    name: 'UI renders line-by-line confidence scores with recognized line text',
    timeoutMs: 105_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'cursive-clean-2lines.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);

      const lineRows = await dom.getLineConfidences();
      assertGreaterOrEqual(lineRows.length, 2, `Expected >= 2 line confidence rows in UI, got ${lineRows.length}`);
      for (const row of lineRows) {
        assert(row.confidencePct !== null && row.confidencePct >= 0 && row.confidencePct <= 100, `Invalid line confidence: ${JSON.stringify(row)}`);
        assert(row.text.length > 0, `Line row missing text: ${JSON.stringify(row)}`);
      }
      const combinedRowsText = lineRows.map((r) => r.text.toLowerCase()).join('\n');
      assertIncludes(combinedRowsText, 'milk', 'Line confidence rows must include recognized token "milk"');
      assertIncludes(combinedRowsText, 'dentist', 'Line confidence rows must include recognized token "dentist"');
    },
  },
  {
    id: 'T1-F7-03',
    tier: 1,
    feature: 'F7',
    name: 'UI renders detected style badge (handwriting / printed)',
    timeoutMs: 105_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'cursive-clean-2lines.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);

      const badge = await dom.getDetectedStyleBadge();
      assert(badge.found && badge.style === 'handwriting', `Expected detected style badge 'handwriting', got ${JSON.stringify(badge)}`);
    },
  },
  {
    id: 'T1-F7-04',
    tier: 1,
    feature: 'F7',
    name: 'Working Copy Text button copies extracted text to clipboard',
    timeoutMs: 105_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'cursive-1line.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);

      const editable = await dom.getEditableText();
      assert(editable.trim().length > 0, 'Expected non-empty editable text');
      const copyRes = await dom.clickCopyText();
      assert(copyRes.clicked, 'Copy Text button not found or not clickable in UI');
      const clip = await dom.readClipboardText();
      assertEqual(clip, editable, 'Clipboard text must match extracted text');
    },
  },
  {
    id: 'T1-F7-05',
    tier: 1,
    feature: 'F7',
    name: 'Copy Text stays synchronized after user edits extracted text',
    timeoutMs: 105_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'cursive-1line.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);

      const customText = 'Edited reminder text: buy 2 gallons of milk';
      await dom.setEditableText(customText);
      const copyRes = await dom.clickCopyText();
      assert(copyRes.clicked, 'Copy Text button not found');
      const clip = await dom.readClipboardText();
      assertEqual(clip, customText, 'Clipboard must receive user-edited text');
    },
  },
  {
    id: 'T1-F7-06',
    tier: 1,
    feature: 'F7',
    name: 'UI metadata badges update cleanly between handwriting and printed scans',
    timeoutMs: 195_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'cursive-1line.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);
      const badge1 = await dom.getDetectedStyleBadge();
      assertEqual(badge1.style, 'handwriting', 'First scan style should be handwriting');

      // Reset page and scan printed
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Printed$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'printed-schedule.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);
      const badge2 = await dom.getDetectedStyleBadge();
      assertEqual(badge2.style, 'printed', 'Second scan style should be printed');
    },
  },

  // =========================================================================
  // F8: Export PNG Consistency, darkInverted & >=90% Accuracy (R5) — 6 Tests
  // =========================================================================
  {
    id: 'T1-F8-01',
    tier: 1,
    feature: 'F8',
    name: 'saveGrayscalePng succeeds after auto-bbox-shrink without VipsImage error',
    timeoutMs: 75_000,
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'server/handwritingRecognizer.ts');
      const procFn = resolveExportFn<(opts: Record<string, unknown>) => Promise<Record<string, any>>>(
        mod,
        ['processImageWithLocalOcr']
      );
      assert(procFn, 'processImageWithLocalOcr missing');
      const buf = fs.readFileSync(path.join(ctx.fixturesDir, 'cursive-wide-margin.png'));
      const res = await procFn({
        imageBuffer: buf,
        ocrProfile: 'handwriting',
        exportStages: true,
        exportRootDir: ctx.createTempDir('ocr-exports-'),
      });
      cleanupExportDir(ctx, res.exportDir);
      assert(res.exportDir && fs.existsSync(res.exportDir), 'exportDir missing');
      const steps: string[] = res.appliedSteps ?? [];
      assert(steps.some((s) => s.startsWith('auto-bbox-shrink(')), `Expected auto-bbox-shrink in steps: ${steps.join(', ')}`);

      const pngs = fs.readdirSync(res.exportDir).filter((f) => f.endsWith('.png'));
      assertGreaterOrEqual(pngs.length, 2, 'Expected exported stage PNGs');
      for (const f of pngs) {
        const { data, info } = await sharp(path.join(res.exportDir, f))
          .raw()
          .toBuffer({ resolveWithObject: true });
        assert(
          info.width > 0 && info.height > 0 && data.length === info.width * info.height * info.channels,
          `Invalid exported PNG pixel buffer ${f}`
        );
      }
    },
  },
  {
    id: 'T1-F8-02',
    tier: 1,
    feature: 'F8',
    name: 'saveGrayscalePng succeeds when both dpi-upscale (>=1.5x) and auto-bbox-shrink trigger',
    timeoutMs: 75_000,
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'server/handwritingRecognizer.ts');
      const procFn = resolveExportFn<(opts: Record<string, unknown>) => Promise<Record<string, any>>>(
        mod,
        ['processImageWithLocalOcr']
      );
      assert(procFn, 'processImageWithLocalOcr missing');
      const buf = fs.readFileSync(path.join(ctx.fixturesDir, 'cursive-wide-margin.png'));
      const res = await procFn({
        imageBuffer: buf,
        ocrProfile: 'handwriting',
        exportStages: true,
        exportRootDir: ctx.createTempDir('ocr-exports-'),
      });
      cleanupExportDir(ctx, res.exportDir);
      const steps: string[] = res.appliedSteps ?? [];
      assertGreaterOrEqual(res.dpiScale ?? 1, 1.5, 'Expected dpiScale >= 1.5');
      assert(steps.some((s) => s.startsWith('dpi-upscale(')), `Expected dpi-upscale step: ${steps.join(', ')}`);
      assert(steps.some((s) => s.startsWith('auto-bbox-shrink(')), `Expected auto-bbox-shrink step: ${steps.join(', ')}`);
      const files = fs.readdirSync(res.exportDir).filter((f) => f.endsWith('.png'));
      for (const f of files) {
        const { data, info } = await sharp(path.join(res.exportDir, f))
          .raw()
          .toBuffer({ resolveWithObject: true });
        assert(
          info.width > 0 && info.height > 0 && data.length === info.width * info.height * info.channels,
          `Corrupt stage PNG pixel buffer ${f}`
        );
      }
    },
  },
  {
    id: 'T1-F8-03',
    tier: 1,
    feature: 'F8',
    name: 'Dark-mode inversion (darkInverted: true) and >= 90% accuracy on dark screenshot',
    timeoutMs: 75_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const fp = path.join(ctx.fixturesDir, 'printed-dark-mode.png');
      const gt = ctx.manifest.fixtures['printed-dark-mode.png'].groundTruth;
      const image = await pngToDataUrl(fp);
      const { status, body } = await srv.postLocalOcr({ image, ocrProfile: 'auto', exportStages: false }, 45_000);
      assertEqual(status, 200, 'HTTP 200 expected');
      assertEqual(body.darkInverted, true, 'darkInverted must be true');
      assert(body.appliedSteps?.includes('color-inversion'), 'appliedSteps must include color-inversion');
      const acc = characterAccuracy(body.text, gt, { caseInsensitive: false });
      assertGreaterOrEqual(acc, 0.9, `Expected >= 0.90 accuracy on dark-mode printed note, got ${acc.toFixed(3)}`);
    },
  },
  {
    id: 'T1-F8-04',
    tier: 1,
    feature: 'F8',
    name: 'Printed benchmark achieves >= 90% (1 - CER) in Offline Local mode',
    timeoutMs: 75_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const fp = path.join(ctx.fixturesDir, 'printed-schedule.png');
      const gt = ctx.manifest.fixtures['printed-schedule.png'].groundTruth;
      const image = await pngToDataUrl(fp);
      const { status, body } = await srv.postLocalOcr({ image, ocrProfile: 'printed', exportStages: false }, 45_000);
      assertEqual(status, 200, 'HTTP 200 expected');
      const acc = characterAccuracy(body.text, gt);
      assertGreaterOrEqual(acc, 0.9, `Expected >= 0.90 printed accuracy, got ${acc.toFixed(3)} ("${body.text}")`);
    },
  },
  {
    id: 'T1-F8-05',
    tier: 1,
    feature: 'F8',
    name: 'Cursive benchmark achieves >= 90% (1 - CER) in Offline Local mode',
    timeoutMs: 75_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const fp = path.join(ctx.fixturesDir, 'cursive-clean-2lines.png');
      const gt = ctx.manifest.fixtures['cursive-clean-2lines.png'].groundTruth;
      const image = await pngToDataUrl(fp);
      const { status, body } = await srv.postLocalOcr({ image, ocrProfile: 'handwriting', exportStages: false }, 55_000);
      assertEqual(status, 200, 'HTTP 200 expected');
      const acc = characterAccuracy(body.text, gt, { caseInsensitive: true });
      assertGreaterOrEqual(acc, 0.9, `Expected >= 0.90 cursive accuracy, got ${acc.toFixed(3)} ("${body.text}")`);
    },
  },
  {
    id: 'T1-F8-06',
    tier: 1,
    feature: 'F8',
    name: 'Vision AI /api/extract-reminder >= 90% (1 - CER) with unified metadata',
    timeoutMs: 110_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      for (const name of ['printed-schedule.png', 'cursive-clean-2lines.png']) {
        const fp = path.join(ctx.fixturesDir, name);
        const gt = ctx.manifest.fixtures[name].groundTruth;
        const image = await pngToDataUrl(fp);
        const { status, body } = await srv.postExtractReminder({ image, ocrProfile: 'auto' }, 45_000);
        assertEqual(status, 200, `${name}: HTTP 200 expected`);
        assertEqual(body.success, true, `${name}: success true`);
        const txt = body.rawText ?? body.ocrText ?? '';
        assert(txt.trim().length > 0, `${name}: rawText/ocrText must be returned`);
        assert(Array.isArray(body.lines) && body.lines.length >= 1, `${name}: unified lines[] metadata required (F8)`);
        assert(
          body.detectedStyle === 'printed' || body.detectedStyle === 'handwriting' || body.detectedStyle === 'mixed',
          `${name}: detectedStyle required (F8), got ${body.detectedStyle}`
        );
        assert(typeof body.accuracyScore === 'number', `${name}: accuracyScore required (F8)`);
        const isPrinted = name.startsWith('printed-');
        const acc = characterAccuracy(txt, gt, { caseInsensitive: !isPrinted });
        assertGreaterOrEqual(acc, 0.9, `${name}: Expected >= 0.90 Vision AI accuracy, got ${acc.toFixed(3)} ("${txt}")`);
      }
    },
  },
];
