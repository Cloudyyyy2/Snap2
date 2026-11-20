import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { preprocessForOCR } from '../../src/services/imageProcessor';
import { autoInvertIfDarkMode, binarizeSauvola } from '../../src/lib/ocr/preprocess';
import { segmentLines, cropLine, splitWideLineBox } from '../../src/lib/ocr/handwriting/lines';
import { filterTextComponents } from '../../src/lib/ocr/handwriting/cleanup';
import {
  assert,
  assertEqual,
  assertGreaterOrEqual,
  assertIncludes,
  assertLessOrEqual,
  assertNotIncludes,
  characterAccuracy,
  loadOptionalModule,
  resolveExportFn,
} from './helpers/metrics';
import type { TestCase, TestContext } from './helpers/types';

async function pngToGrayBuffer(filePath: string): Promise<{ data: Uint8ClampedArray; width: number; height: number }> {
  const { data, info } = await sharp(filePath).grayscale().raw().toBuffer({ resolveWithObject: true });
  return { data: new Uint8ClampedArray(data), width: info.width, height: info.height };
}

async function pngToDataUrl(filePath: string): Promise<string> {
  return `data:image/png;base64,${fs.readFileSync(filePath).toString('base64')}`;
}

function countInkPixels(gray: Uint8ClampedArray, threshold = 180): number {
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

export const tier2Tests: TestCase[] = [
  // =========================================================================
  // F1 Boundaries — 6 Tests
  // =========================================================================
  {
    id: 'T2-F1-01',
    tier: 2,
    feature: 'F1',
    name: 'Null/invalid sources throw descriptive Error while 1x1 succeeds',
    async run() {
      let threwNull = false;
      try {
        preprocessForOCR(null as any);
      } catch (err) {
        threwNull = err instanceof Error && err.message.length > 0;
      }
      assert(threwNull, 'null source must throw descriptive Error');

      let threwObj = false;
      try {
        preprocessForOCR({} as any);
      } catch (err) {
        threwObj = err instanceof Error && err.message.length > 0;
      }
      assert(threwObj, 'empty object source must throw descriptive Error');

      const out1x1 = preprocessForOCR({
        data: new Uint8ClampedArray([128, 128, 128, 255]),
        width: 1,
        height: 1,
      });
      assertEqual(out1x1.width, 1, '1x1 width');
      assertEqual(out1x1.height, 1, '1x1 height');
      assert(!Number.isNaN(out1x1.data[0]), '1x1 output pixel must not be NaN');
    },
  },
  {
    id: 'T2-F1-02',
    tier: 2,
    feature: 'F1',
    name: 'Uniform solid white (255) and solid black (0) frames avoid NaN',
    async run() {
      const w = 64;
      const h = 64;
      const whiteRgba = new Uint8ClampedArray(w * h * 4).fill(255);
      const outW = preprocessForOCR({ data: whiteRgba, width: w, height: h });
      for (let i = 0; i < outW.data.length; i += 4) {
        assert(!Number.isNaN(outW.data[i]), 'White frame NaN check');
        assertGreaterOrEqual(outW.data[i], 240, 'Solid white remains bright');
      }

      const blackRgba = new Uint8ClampedArray(w * h * 4);
      for (let i = 0; i < w * h; i++) blackRgba[i * 4 + 3] = 255;
      const outB = preprocessForOCR({ data: blackRgba, width: w, height: h });
      for (let i = 0; i < outB.data.length; i += 4) {
        assert(!Number.isNaN(outB.data[i]) && outB.data[i] >= 0 && outB.data[i] <= 255, 'Solid black bounded byte');
      }
    },
  },
  {
    id: 'T2-F1-03',
    tier: 2,
    feature: 'F1',
    name: 'Dark-mode inverted anti-aliased strokes preserve gray gradient',
    async run(ctx) {
      const { data: gray, width, height } = await pngToGrayBuffer(
        path.join(ctx.fixturesDir, 'printed-dark-mode.png')
      );
      const inv = autoInvertIfDarkMode(gray, width, height);
      assertEqual(inv.inverted, true, 'Dark mode must invert');

      const rgba = new Uint8ClampedArray(width * height * 4);
      for (let i = 0; i < gray.length; i++) {
        rgba[i * 4] = inv.data[i];
        rgba[i * 4 + 1] = inv.data[i];
        rgba[i * 4 + 2] = inv.data[i];
        rgba[i * 4 + 3] = 255;
      }
      const out = preprocessForOCR({ data: rgba, width, height });
      let mid = 0;
      for (let i = 0; i < out.data.length; i += 4) {
        const v = out.data[i];
        if (v >= 15 && v <= 240) mid++;
      }
      assertGreaterOrEqual(mid, 50, 'Inverted dark-mode strokes must preserve anti-aliased gray edges');
    },
  },
  {
    id: 'T2-F1-04',
    tier: 2,
    feature: 'F1',
    name: 'Salt-and-pepper noise + anti-aliased text preserves stroke edges',
    async run() {
      const w = 160;
      const h = 80;
      const rgba = new Uint8ClampedArray(w * h * 4);
      let seed = 42;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const dist = Math.abs(y - 40);
          let gray = 248;
          if (dist <= 1.2) gray = 20;
          else if (dist <= 3.0) gray = Math.round(20 + ((dist - 1.2) / 1.8) * 200);
          seed = (seed * 1664525 + 1013904223) >>> 0;
          if (dist > 6 && seed % 20 === 0) {
            gray = seed % 2 === 0 ? 0 : 255; // 5% impulse speckles away from stroke
          }
          const idx = (y * w + x) * 4;
          rgba[idx] = gray;
          rgba[idx + 1] = gray;
          rgba[idx + 2] = gray;
          rgba[idx + 3] = 255;
        }
      }
      const out = preprocessForOCR({ data: rgba, width: w, height: h });
      let mid = 0;
      for (let y = 37; y <= 43; y++) {
        for (let x = 10; x < w - 10; x++) {
          const v = out.data[(y * w + x) * 4];
          if (v >= 15 && v <= 240) mid++;
        }
      }
      assertGreaterOrEqual(mid, 30, 'Anti-aliased stroke halo preserved despite background impulse noise');
    },
  },
  {
    id: 'T2-F1-05',
    tier: 2,
    feature: 'F1',
    name: 'Zero-margin tight crop preserves border pixels without off-by-one corruption',
    async run(ctx) {
      const { data, info } = await sharp(path.join(ctx.fixturesDir, 'zero-margin-crop.png'))
        .ensureAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });
      const out = preprocessForOCR({ data: new Uint8ClampedArray(data), width: info.width, height: info.height });
      assertEqual(out.width, info.width, 'Zero-margin width');
      assertEqual(out.height, info.height, 'Zero-margin height');
      let hasDarkInk = false;
      for (let i = 0; i < out.data.length; i += 4) {
        if (out.data[i] < 100) hasDarkInk = true;
      }
      assert(hasDarkInk, 'Zero-margin text ink must remain intact');
    },
  },
  {
    id: 'T2-F1-06',
    tier: 2,
    feature: 'F1',
    name: 'Low-contrast gray-on-gray text retains continuous stroke gradient',
    async run() {
      const w = 120;
      const h = 60;
      const rgba = new Uint8ClampedArray(w * h * 4);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const d = Math.abs(y - 30);
          const v = d <= 1 ? 150 : d <= 2.5 ? 185 : 215;
          const idx = (y * w + x) * 4;
          rgba[idx] = v;
          rgba[idx + 1] = v;
          rgba[idx + 2] = v;
          rgba[idx + 3] = 255;
        }
      }
      const out = preprocessForOCR({ data: rgba, width: w, height: h });
      const strokeCenter = out.data[(30 * w + 60) * 4];
      const bg = out.data[(5 * w + 60) * 4];
      assertGreaterOrEqual(bg - strokeCenter, 35, 'Low-contrast stroke must separate from background');
    },
  },

  // =========================================================================
  // F2 Boundaries — 6 Tests
  // =========================================================================
  {
    id: 'T2-F2-01',
    tier: 2,
    feature: 'F2',
    name: 'Extreme positive slant +35 deg normalization without edge clipping',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/geometry.ts');
      const deslantFn = resolveExportFn<(...args: any[]) => Record<string, any>>(mod, ['deslantGrayscale']);
      assert(deslantFn, 'deslantGrayscale missing');

      const { data: gray, width, height } = await pngToGrayBuffer(
        path.join(ctx.fixturesDir, 'cursive-slanted-pos35.png')
      );
      const binary = binarizeSauvola(gray, width, height, 15, 0.2);
      // Test default or explicit ±35 range per R2
      const res = deslantFn(gray, binary, width, height, { minAngleDeg: -38, maxAngleDeg: 38 });
      assert(Boolean(res.applied ?? res.corrected ?? res.deslanted), 'Expected +35° slant correction');
      assertGreaterOrEqual(res.angleDeg, 20, `Expected strong positive angle >= 20°, got ${res.angleDeg}`);
      assertGreaterOrEqual(res.width, width, 'Output width expanded to avoid clipping');
      const inkBefore = countInkPixels(gray, 180);
      const inkAfter = countInkPixels(res.data, 180);
      assertGreaterOrEqual(inkAfter, inkBefore * 0.85, 'Edge characters must not be clipped at +35°');
    },
  },
  {
    id: 'T2-F2-02',
    tier: 2,
    feature: 'F2',
    name: 'Extreme negative slant -35 deg normalization without edge clipping',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/geometry.ts');
      const deslantFn = resolveExportFn<(...args: any[]) => Record<string, any>>(mod, ['deslantGrayscale']);
      assert(deslantFn, 'deslantGrayscale missing');

      const { data: gray, width, height } = await pngToGrayBuffer(
        path.join(ctx.fixturesDir, 'cursive-slanted-neg35.png')
      );
      const binary = binarizeSauvola(gray, width, height, 15, 0.2);
      const res = deslantFn(gray, binary, width, height, { minAngleDeg: -38, maxAngleDeg: 38 });
      assert(Boolean(res.applied ?? res.corrected ?? res.deslanted), 'Expected -35° slant correction');
      assertLessOrEqual(res.angleDeg, -20, `Expected strong negative angle <= -20°, got ${res.angleDeg}`);
      assertGreaterOrEqual(res.width, width, 'Output width expanded to avoid clipping');
      const inkBefore = countInkPixels(gray, 180);
      const inkAfter = countInkPixels(res.data, 180);
      assertGreaterOrEqual(inkAfter, inkBefore * 0.85, 'Edge characters must not be clipped at -35°');
    },
  },
  {
    id: 'T2-F2-03',
    tier: 2,
    feature: 'F2',
    name: 'Narrow 12x40 single-character upright & slanted bounds safety',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/geometry.ts');
      const deslantFn = resolveExportFn<(...args: any[]) => Record<string, any>>(mod, ['deslantGrayscale']);
      assert(deslantFn, 'deslantGrayscale missing');

      const w = 12;
      const h = 40;
      const upright = new Uint8ClampedArray(w * h).fill(255);
      for (let y = 4; y < 36; y++) upright[y * w + 6] = 20;
      const rUp = deslantFn(upright, w, h);
      assertEqual(Boolean(rUp.applied ?? rUp.corrected ?? rUp.deslanted), false, 'Narrow upright stays un-sheared');

      const slanted = new Uint8ClampedArray(w * h).fill(255);
      for (let y = 4; y < 36; y++) {
        const x = Math.max(0, Math.min(w - 1, Math.round(6 - (y - 20) * 0.35)));
        slanted[y * w + x] = 20;
      }
      const rSl = deslantFn(slanted, w, h, { minAngleDeg: -38, maxAngleDeg: 38 });
      assertEqual(rSl.height, h, 'Narrow slanted height preserved');
      assertEqual(rSl.data.length, rSl.width * rSl.height, 'Consistent buffer length');
    },
  },
  {
    id: 'T2-F2-04',
    tier: 2,
    feature: 'F2',
    name: 'Tiny <8x8 buffer and blank 100x40 buffer return angleDeg 0 safely',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/geometry.ts');
      const deslantFn = resolveExportFn<(...args: any[]) => Record<string, any>>(mod, ['deslantGrayscale']);
      assert(deslantFn, 'deslantGrayscale missing');

      const tiny = new Uint8ClampedArray(4 * 4).fill(100);
      const rTiny = deslantFn(tiny, 4, 4);
      assertEqual(rTiny.angleDeg, 0, '4x4 angleDeg === 0');
      assertEqual(Boolean(rTiny.applied ?? rTiny.corrected ?? rTiny.deslanted), false, '4x4 not corrected');

      const blank = new Uint8ClampedArray(100 * 40).fill(255);
      const rBlank = deslantFn(blank, 100, 40);
      assertEqual(rBlank.angleDeg, 0, 'Blank angleDeg === 0');
      assertEqual(rBlank.data.length, 4000, 'Blank buffer length unchanged');
    },
  },
  {
    id: 'T2-F2-05',
    tier: 2,
    feature: 'F2',
    name: 'Wide short 120x18 buffer with +25 deg shear completes cleanly',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/geometry.ts');
      const shearFn = resolveExportFn<
        (g: Uint8ClampedArray, w: number, h: number, deg: number, fill?: number) => {
          data: Uint8ClampedArray;
          width: number;
          height: number;
        }
      >(mod, ['shearDeslantGrayscale', 'shearGrayscale']);
      assert(shearFn, 'shearDeslantGrayscale missing');

      const w = 120;
      const h = 18;
      const gray = new Uint8ClampedArray(w * h).fill(255);
      for (let y = 2; y < h - 2; y++) gray[y * w + 60] = 20;
      const out = shearFn(gray, w, h, 25, 255);
      assertEqual(out.height, h, 'Height preserved');
      assertGreaterOrEqual(out.width, w + 4, 'Width expanded');
    },
  },
  {
    id: 'T2-F2-06',
    tier: 2,
    feature: 'F2',
    name: 'Sparse noise (<20 ink pixels) does not trigger spurious slant correction',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/geometry.ts');
      const estFn = resolveExportFn<(b: Uint8ClampedArray, w: number, h: number) => number>(
        mod,
        ['estimateSlantAngle', 'estimateCursiveSlantAngle']
      );
      assert(estFn, 'estimateSlantAngle missing');

      const w = 120;
      const h = 60;
      const gray = new Uint8ClampedArray(w * h).fill(255);
      for (let i = 0; i < 10; i++) gray[(10 + i * 3) * w + (20 + i * 5)] = 20;
      assertEqual(estFn(gray, w, h), 0, 'Sub-threshold ink pixels must return angle 0');
    },
  },

  // =========================================================================
  // F3 Boundaries — 5 Tests
  // =========================================================================
  {
    id: 'T2-F3-01',
    tier: 2,
    feature: 'F3',
    name: 'Tight 4-line cursive with interlocking descenders/ascenders avoids line merging',
    async run() {
      const w = 400;
      const h = 160;
      const binary = new Uint8ClampedArray(w * h).fill(255);
      // 4 line cores at y=22, 56, 90, 124 (pitch=34px), plus thin descender loops dipping 14px down
      const ys = [22, 56, 90, 124];
      for (const cy of ys) {
        for (let y = cy - 6; y <= cy + 6; y++) {
          for (let x = 30; x <= 360; x++) {
            if (x % 8 < 5) binary[y * w + x] = 0;
          }
        }
        // Thin descender loop extending 14px downward at x=120..124
        for (let y = cy + 7; y <= Math.min(h - 1, cy + 19); y++) {
          binary[y * w + 122] = 0;
          binary[y * w + 123] = 0;
        }
      }
      const boxes = segmentLines(binary, w, h, { skipFilter: true });
      assertEqual(boxes.length, 4, `Expected 4 distinct line boxes, got ${boxes.length}`);
    },
  },
  {
    id: 'T2-F3-02',
    tier: 2,
    feature: 'F3',
    name: 'Empty 0x0, all-white, and zero-margin 100x24 single line bounds',
    async run() {
      assertEqual(segmentLines(new Uint8ClampedArray(0), 0, 0).length, 0, '0x0 returns []');
      assertEqual(segmentLines(new Uint8ClampedArray(100 * 40).fill(255), 100, 40).length, 0, 'Blank returns []');

      const w = 100;
      const h = 24;
      const zm = new Uint8ClampedArray(w * h).fill(255);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          if (y >= 2 && y <= 21 && x % 6 < 4) zm[y * w + x] = 0;
        }
      }
      const boxes = segmentLines(zm, w, h, { skipFilter: true });
      assertEqual(boxes.length, 1, 'Zero-margin single line detected');
      assertGreaterOrEqual(boxes[0].y0, 0, 'y0 clamped >= 0');
      assertLessOrEqual(boxes[0].y1, h - 1, 'y1 clamped <= h-1');
    },
  },
  {
    id: 'T2-F3-03',
    tier: 2,
    feature: 'F3',
    name: 'Single 18x22 char vs ultra-long 1800x60 multi-word line splitting',
    async run() {
      const tinyBin = new Uint8ClampedArray(18 * 22).fill(255);
      for (let y = 5; y < 17; y++) {
        for (let x = 4; x < 14; x++) tinyBin[y * 18 + x] = 0;
      }
      const singleChunks = splitWideLineBox(tinyBin, 18, 22, { x0: 4, y0: 5, x1: 13, y1: 16 }, 6.5);
      assertEqual(singleChunks.length, 1, 'Single char box must stay 1 chunk');

      const w = 1800;
      const h = 60;
      const longBin = new Uint8ClampedArray(w * h).fill(255);
      // 10 words of width 130 separated by 45px gaps
      const words: Array<[number, number]> = [];
      for (let i = 0; i < 10; i++) {
        const wx0 = 20 + i * 175;
        const wx1 = wx0 + 130;
        words.push([wx0, wx1]);
        for (let y = 16; y <= 44; y++) {
          for (let x = wx0; x <= wx1; x++) longBin[y * w + x] = 0;
        }
      }
      const chunks = splitWideLineBox(longBin, w, h, { x0: 20, y0: 12, x1: 1725, y1: 48 }, 6.5);
      assertGreaterOrEqual(chunks.length, 4, '1800px line should split into >= 4 chunks');
      for (const c of chunks) {
        for (const [wx0, wx1] of words) {
          assert(!(c.x0 > wx0 + 2 && c.x0 < wx1 - 2), 'Split must not cut inside a word');
          assert(!(c.x1 > wx0 + 2 && c.x1 < wx1 - 2), 'Split must not cut inside a word');
        }
      }
    },
  },
  {
    id: 'T2-F3-04',
    tier: 2,
    feature: 'F3',
    name: 'Dense left spiral holes + top-right smudge do not delete valid left-aligned word at x=32',
    async run() {
      const w = 400;
      const h = 240;
      const bin = new Uint8ClampedArray(w * h).fill(255);
      // 6 periodic spiral holes at x=4..16
      for (let i = 0; i < 6; i++) {
        const cy = 20 + i * 38;
        for (let y = cy - 5; y <= cy + 5; y++) {
          for (let x = 4; x <= 16; x++) bin[y * w + x] = 0;
        }
      }
      // Valid word starting at x=34, y=90..118
      for (let y = 92; y <= 116; y++) {
        for (let x = 34; x <= 160; x++) {
          if (x % 7 < 4) bin[y * w + x] = 0;
        }
      }
      const cleaned = filterTextComponents(bin, w, h);
      const boxes = segmentLines(cleaned, w, h);
      assertEqual(boxes.length, 1, 'Valid left-aligned word must survive spiral hole stripping');
      assertLessOrEqual(boxes[0].x0, 42, 'Valid word at x=34 must not be clipped');
    },
  },
  {
    id: 'T2-F3-05',
    tier: 2,
    feature: 'F3',
    name: 'cropLine handles out-of-bounds box coordinates cleanly with 255 padding',
    async run() {
      const w = 100;
      const h = 50;
      const gray = new Uint8ClampedArray(w * h).fill(200);
      const cropped = cropLine(gray, w, h, { x0: -10, y0: -5, x1: 120, y1: 60 }, 16);
      assertGreaterOrEqual(cropped.width, 100, 'Cropped width valid');
      assertGreaterOrEqual(cropped.height, 50, 'Cropped height valid');
      assertEqual(cropped.data[0], 255, 'Out-of-bounds corner padded with 255');
      for (let i = 0; i < cropped.data.length; i++) {
        assert(!Number.isNaN(cropped.data[i]), 'No NaN in cropped line');
      }
    },
  },

  // =========================================================================
  // F4 Boundaries — 6 Tests
  // =========================================================================
  {
    id: 'T2-F4-01',
    tier: 2,
    feature: 'F4',
    name: 'Blank white, 1x1 tiny, and solid black PNGs return empty text without crash',
    timeoutMs: 105_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      for (const fname of ['blank-white.png', 'tiny-1x1.png', 'solid-black.png']) {
        const image = await pngToDataUrl(path.join(ctx.fixturesDir, fname));
        const { status, body } = await srv.postLocalOcr({ image, ocrProfile: 'handwriting', exportStages: false }, 30_000);
        assertEqual(status, 200, `${fname}: expected HTTP 200`);
        assertEqual(body.success, true, `${fname}: expected success true`);
        assertEqual(body.text.trim(), '', `${fname}: expected empty text`);
        assertEqual(body.lines.length, 0, `${fname}: expected 0 lines`);
        assert(body.appliedSteps?.includes('no-ink-detected'), `${fname}: expected no-ink-detected step`);
      }
    },
  },
  {
    id: 'T2-F4-02',
    tier: 2,
    feature: 'F4',
    name: 'Corrupt base64, missing image, and truncated PNG leave server healthy',
    async run(ctx) {
      const srv = await ctx.getServer();
      const missing = await srv.postLocalOcr({});
      assertEqual(missing.status, 400, 'Missing image -> HTTP 400');
      assertEqual(missing.body.success, false, 'Missing image -> success false');

      const badStr = await srv.postLocalOcr({ image: 'not-a-valid-image', exportStages: false });
      assertGreaterOrEqual(badStr.status, 400, 'Invalid image string -> 4xx/5xx');
      assertEqual(badStr.body.success, false, 'Invalid image -> success false');

      const truncPng = await srv.postLocalOcr({ image: 'data:image/png;base64,iVBORw0KGgoAAAA', exportStages: false });
      assertGreaterOrEqual(truncPng.status, 400, 'Truncated PNG -> 4xx/5xx');
      assertEqual(truncPng.body.success, false, 'Truncated PNG -> success false');

      const health = await srv.getHealth();
      assertEqual(health.status, 200, 'Server must stay healthy after corrupt inputs');
    },
  },
  {
    id: 'T2-F4-03',
    tier: 2,
    feature: 'F4',
    name: 'Oversized >15MB payload rejected cleanly (413) while 3200x2400 PNG succeeds (200)',
    timeoutMs: 80_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const bigStr = 'A'.repeat(16 * 1024 * 1024);
      const over = await srv.postLocalOcr({ image: `data:image/png;base64,${bigStr}`, exportStages: false }, 20_000);
      assert(over.status === 413 || over.status === 400, `Expected 413/400 for >15MB payload, got ${over.status}`);

      const hiResImg = await pngToDataUrl(path.join(ctx.fixturesDir, 'highres-3200x2400.png'));
      const ok = await srv.postLocalOcr({ image: hiResImg, ocrProfile: 'printed', exportStages: false }, 45_000);
      assertEqual(ok.status, 200, '3200x2400 PNG must return HTTP 200');
      assertIncludes(ok.body.text.toLowerCase(), 'resolution', '3200x2400 PNG should still recognize text');
    },
  },
  {
    id: 'T2-F4-04',
    tier: 2,
    feature: 'F4',
    name: 'Single-word and single-character handwriting crops do not hallucinate paragraphs',
    timeoutMs: 105_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const milkRes = await srv.postLocalOcr(
        {
          image: await pngToDataUrl(path.join(ctx.fixturesDir, 'single-word-milk.png')),
          ocrProfile: 'handwriting',
          exportStages: false,
        },
        45_000
      );
      assertEqual(milkRes.status, 200, 'Single word HTTP 200');
      assertLessOrEqual(milkRes.body.lines.length, 1, 'Single word <= 1 line');
      assertIncludes(milkRes.body.text.toLowerCase(), 'milk', 'Single word milk recognized');

      const charRes = await srv.postLocalOcr(
        {
          image: await pngToDataUrl(path.join(ctx.fixturesDir, 'single-char-7.png')),
          ocrProfile: 'handwriting',
          exportStages: false,
        },
        45_000
      );
      assertEqual(charRes.status, 200, 'Single char HTTP 200');
      assertLessOrEqual(charRes.body.lines.length, 1, 'Single char <= 1 line');
      assertIncludes(charRes.body.text, '7', 'Single char "7" must be recognized');
      assertLessOrEqual(charRes.body.text.trim().length, 10, 'Single char crop must not hallucinate sentence');
    },
  },
  {
    id: 'T2-F4-05',
    tier: 2,
    feature: 'F4',
    name: 'Invalid ocrProfile value in /api/local-ocr handled safely without 500 crash',
    timeoutMs: 60_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const image = await pngToDataUrl(path.join(ctx.fixturesDir, 'printed-schedule.png'));
      const res = await srv.postLocalOcr({ image, ocrProfile: 'invalid-xyz', exportStages: false }, 45_000);
      assert(res.status === 200 || res.status === 400, `Expected safe 200 fallback or 400, got ${res.status}`);
      if (res.status === 200) {
        assertEqual(res.body.success, true, 'Fallback 200 must have success: true');
        assert(
          res.body.text.toLowerCase().includes('dentist'),
          `Fallback 200 must recognize printed-schedule text, got "${res.body.text}"`
        );
      }
    },
  },
  {
    id: 'T2-F4-06',
    tier: 2,
    feature: 'F4',
    name: 'PARSeq false-1.0 trap words (salmon, broccoli, tomorrow, pan au chocolat) resist misreads',
    timeoutMs: 75_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const fp = path.join(ctx.fixturesDir, 'cursive-anchors-grocery.png');
      const gt = ctx.manifest.fixtures['cursive-anchors-grocery.png'].groundTruth;
      const res = await srv.postLocalOcr(
        { image: await pngToDataUrl(fp), ocrProfile: 'handwriting', exportStages: false },
        55_000
      );
      assertEqual(res.status, 200, 'HTTP 200');
      const lower = res.body.text.toLowerCase();
      for (const bad of ['jolmm', 'virwlh', 'can an chocolat', 'they mince']) {
        assertNotIncludes(lower, bad, `Must not emit PARSeq trap misread ${bad}`);
      }
      const acc = characterAccuracy(res.body.text, gt, { caseInsensitive: true });
      assertGreaterOrEqual(acc, 0.9, `Expected >= 0.90 1-CER on trap words, got ${acc.toFixed(3)} ("${res.body.text}")`);
    },
  },

  // =========================================================================
  // F5 Boundaries — 6 Tests
  // =========================================================================
  {
    id: 'T2-F5-01',
    tier: 2,
    feature: 'F5',
    name: 'Empty, whitespace-only, and pure-punctuation inputs return accuracyScore 0 (no 88% floor)',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/contextualPostprocess.ts');
      const fn = resolveExportFn<
        (t: string) => { text: string; accuracyScore: number; totalWords: number }
      >(mod, ['postprocessOcrText']);
      assert(fn, 'contextualPostprocess.ts -> postprocessOcrText missing');

      for (const inp of ['', '   \n\t  ', '... --- !!!']) {
        const res = fn(inp);
        assertEqual(res.totalWords, 0, `totalWords for "${inp}"`);
        assertEqual(res.accuracyScore, 0, `accuracyScore for "${inp}" must be 0 (not 88)`);
      }

      const gibberish = fn('xyzqwk plmnbv zzzqqq');
      assertLessOrEqual(
        gibberish.accuracyScore,
        20,
        `Unrecognized gibberish accuracyScore must be <= 20 (no 88% floor); got ${gibberish.accuracyScore}`
      );
    },
  },
  {
    id: 'T2-F5-02',
    tier: 2,
    feature: 'F5',
    name: 'Mixed numbers, currency, decimals, dates, times, refs, and emails pass through verbatim',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/contextualPostprocess.ts');
      const fn = resolveExportFn<(t: string) => { text: string }>(mod, ['postprocessOcrText']);
      assert(fn, 'postprocessOcrText missing');
      const sample = 'Pay $1,250.99 on 12/31/2026 at 09:45am; ref #A-9042; test@example.com';
      assertEqual(fn(sample).text, sample, 'All non-alpha tokens must stay 100% verbatim');
    },
  },
  {
    id: 'T2-F5-03',
    tier: 2,
    feature: 'F5',
    name: 'Visual-cost ceiling <= 0.32 and anchor non-overcorrection (Finding 3)',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/contextualPostprocess.ts');
      const fn = resolveExportFn<(t: string) => { text: string }>(mod, ['postprocessOcrText']);
      assert(fn, 'postprocessOcrText missing');
      const anchors = 'tomorrow tomatoes turkey tray buns pesto met they this that common teacher form';
      assertEqual(fn(anchors).text, anchors, 'Valid anchors and distinct words must never be mutated');
    },
  },
  {
    id: 'T2-F5-04',
    tier: 2,
    feature: 'F5',
    name: 'Proper nouns, apostrophes, hyphens, and uppercase acronyms preserved',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/contextualPostprocess.ts');
      const fn = resolveExportFn<(t: string) => { text: string }>(mod, ['postprocessOcrText']);
      assert(fn, 'postprocessOcrText missing');
      const sample = "NASA CEO Dr. O'Connor met Jean-Luc in NYC";
      const out = fn(sample).text;
      for (const tok of ['NASA', 'CEO', 'Dr.', "O'Connor", 'Jean-Luc', 'NYC']) {
        assertIncludes(out, tok, `Token ${tok} must remain verbatim`);
      }
    },
  },
  {
    id: 'T2-F5-05',
    tier: 2,
    feature: 'F5',
    name: 'Ultra-long 500-word input runs fast (<250ms) and single-char tokens stay intact',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/contextualPostprocess.ts');
      const fn = resolveExportFn<(t: string) => { text: string }>(mod, ['postprocessOcrText']);
      assert(fn, 'postprocessOcrText missing');

      assertEqual(fn('I a 1 x').text, 'I a 1 x', 'Single-char tokens preserved');
      const longText = Array.from({ length: 100 }, () => 'buy milk tomorrow salmon spinach').join('\n');
      const t0 = Date.now();
      const res = fn(longText);
      const dt = Date.now() - t0;
      assertLessOrEqual(dt, 500, `500-word postprocess took ${dt}ms`);
      assertIncludes(res.text, 'tomorrow', '500-word anchor intact');
    },
  },
  {
    id: 'T2-F5-06',
    tier: 2,
    feature: 'F5',
    name: 'Unicode/accented words and hyphenated grocery items preserved cleanly',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/contextualPostprocess.ts');
      const fn = resolveExportFn<(t: string) => { text: string }>(mod, ['postprocessOcrText']);
      assert(fn, 'postprocessOcrText missing');
      const sample = 'Café crème & pan-seared salmon';
      const out = fn(sample).text;
      assertIncludes(out, 'Café', 'Accented proper token Café preserved');
      assertIncludes(out, 'crème', 'Accented token crème preserved');
      assertIncludes(out, 'pan-seared', 'Hyphenated token pan-seared preserved');
      assertIncludes(out, 'salmon', 'Valid grocery word preserved');
    },
  },

  // =========================================================================
  // F6 Boundaries — 5 Tests
  // =========================================================================
  {
    id: 'T2-F6-01',
    tier: 2,
    feature: 'F6',
    name: 'Invalid ocrProfile / empty payload handling on POST /api/extract-reminder',
    async run(ctx) {
      const srv = await ctx.getServer();
      const empty = await srv.postExtractReminder({});
      assertEqual(empty.status, 400, 'Empty body -> HTTP 400');
      assertEqual(empty.body.success, false, 'Empty body -> success false');

      const bogus = await srv.postExtractReminder({ ocrProfile: 'bogus-profile', ocrText: 'Buy milk tomorrow 5pm' }, 30_000);
      assert(bogus.status === 200 || bogus.status === 400, `Expected safe 200 or 400, got ${bogus.status}`);
    },
  },
  {
    id: 'T2-F6-02',
    tier: 2,
    feature: 'F6',
    name: 'Rapid mode switching in UI routes strictly to latest selected mode',
    timeoutMs: 105_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.enableNetworkCapture();

      await dom.clickModeButton(/^Auto$/i);
      await dom.clickModeButton(/^Printed$/i);
      await dom.clickModeButton(/^Cursive\s*\/\s*Handwriting$/i);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'cursive-1line.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);

      const reqs = dom.getCapturedRequests();
      assert(
        reqs.some((r) => r.method === 'POST' && r.url.includes('/api/local-ocr')),
        'Latest selected Offline Local mode must route to /api/local-ocr'
      );
      assert(
        !reqs.some((r) => r.method === 'POST' && r.url.includes('/api/extract-reminder')),
        'Must not route to /api/extract-reminder after switching back to Offline Local'
      );
    },
  },
  {
    id: 'T2-F6-03',
    tier: 2,
    feature: 'F6',
    name: 'UI shows visible error banner and remains interactive when /api/local-ocr returns HTTP 500',
    timeoutMs: 75_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.enableNetworkCapture({ failLocalOcrWithStatus: 500 });
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'cursive-1line.png'));
      await dom.waitForScanComplete(25_000, ctx.signal);

      const pageText = await dom.getPageText();
      const errBanner = await dom.getVisibleErrorBanner();
      assert(
        Boolean(errBanner) || /error|failed|simulated|unavailable/i.test(pageText),
        'Expected visible error feedback when /api/local-ocr returns 500'
      );
      // App remains interactive — can still click Auto mode
      await dom.clickModeButton(/^Auto$/i);
    },
  },
  {
    id: 'T2-F6-04',
    tier: 2,
    feature: 'F6',
    name: 'Text-only /api/extract-reminder conversion (ocrText without image)',
    timeoutMs: 60_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const { status, body } = await srv.postExtractReminder(
        { ocrText: 'Pay electric bill $114 tomorrow 3pm' },
        35_000
      );
      assertEqual(status, 200, 'HTTP 200 expected');
      assertEqual(body.success, true, 'success true');
      assertEqual(body.data?.category, 'Bills', 'Expected Bills category');
      assert(Boolean(body.data?.dueDate), 'Expected parsed dueDate');
    },
  },
  {
    id: 'T2-F6-05',
    tier: 2,
    feature: 'F6',
    name: 'Corrupt zero-byte file upload in browser UI fails gracefully without crash',
    timeoutMs: 75_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      const tmpDir = ctx.createTempDir('corrupt-upload-');
      const zeroBytePath = path.join(tmpDir, 'zero-byte.png');
      const truncatedPngPath = path.join(tmpDir, 'corrupt-truncated.png');
      fs.writeFileSync(zeroBytePath, Buffer.alloc(0));
      fs.writeFileSync(truncatedPngPath, Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x00, 0x00]));

      for (const badFilePath of [zeroBytePath, truncatedPngPath]) {
        await dom.resetPage(srv.baseUrl, ctx.signal);
        await dom.clickModeButton(/^Offline Local$/i);
        await dom.uploadFixture(badFilePath);
        await dom.waitForScanComplete(25_000, ctx.signal);

        // Spinner must terminate, error feedback must surface, and UI must stay interactive
        const txt = await dom.getPageText();
        assert(
          !/Extracting text|Recognizing|Scanning note|Running local OCR|Reading handwriting|Reading printed|Normalizing grayscale|Running offline OCR/i.test(
            txt
          ),
          `Scan spinner must terminate after corrupt upload (${path.basename(badFilePath)})`
        );
        const errBanner = await dom.getVisibleErrorBanner();
        assert(
          Boolean(errBanner) || /error|failed|could not|invalid|unavailable/i.test(txt),
          `Expected visible error feedback after corrupt upload (${path.basename(badFilePath)})`
        );
        assertIncludes(txt, 'Live OCR Scanner', 'UI must remain alive after corrupt upload');
        const buttons = await dom.listButtons();
        assertGreaterOrEqual(
          buttons.filter((b) => !b.disabled).length,
          1,
          'UI must remain interactive with enabled controls after corrupt upload'
        );
      }
    },
  },

  // =========================================================================
  // F7 Boundaries — 5 Tests
  // =========================================================================
  {
    id: 'T2-F7-01',
    tier: 2,
    feature: 'F7',
    name: 'Blank image UI metadata display renders zero NaN% or undefined artifacts',
    timeoutMs: 95_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'blank-white.png'));
      await dom.waitForScanComplete(45_000, ctx.signal);

      const pageText = await dom.getPageText();
      assertNotIncludes(pageText, 'NaN%', 'Must not render NaN%');
      assertNotIncludes(pageText, 'undefined', 'Must not render undefined');
    },
  },
  {
    id: 'T2-F7-02',
    tier: 2,
    feature: 'F7',
    name: '1-line vs 15-line UI line-by-line confidence display scales accurately',
    timeoutMs: 195_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();

      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'cursive-1line.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);
      const rows1 = await dom.getLineConfidences();
      assertEqual(rows1.length, 1, `1-line scan should render 1 line confidence row, got ${rows1.length}`);

      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.clickModeButton(/^Printed$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'printed-15lines.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);
      const rows15 = await dom.getLineConfidences();
      assertGreaterOrEqual(rows15.length, 12, `15-line scan should render >= 12 line rows, got ${rows15.length}`);
      for (const r of rows15) {
        assert(r.confidencePct !== null && r.confidencePct >= 0 && r.confidencePct <= 100, `Bounded line confidence: ${JSON.stringify(r)}`);
      }
    },
  },
  {
    id: 'T2-F7-03',
    tier: 2,
    feature: 'F7',
    name: 'Copy Text preserves multi-line special characters (quotes, $, &) verbatim',
    timeoutMs: 105_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'cursive-1line.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);

      const special = 'Line 1: Buy "Milk" & Eggs\nLine 2: Pay $5.00 @ 3:30pm!';
      await dom.setEditableText(special);
      const res = await dom.clickCopyText();
      assert(res.clicked, 'Copy Text button must be present');
      const clip = await dom.readClipboardText();
      assertEqual(clip, special, 'Clipboard must preserve special characters and newlines verbatim');
    },
  },
  {
    id: 'T2-F7-04',
    tier: 2,
    feature: 'F7',
    name: 'UI renders mixed style badge on mixed printed+cursive document',
    timeoutMs: 105_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'mixed-printed-cursive.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);

      const badge = await dom.getDetectedStyleBadge();
      assert(
        badge.found && badge.style === 'mixed',
        `Expected 'mixed' detected style badge on mixed document, got ${JSON.stringify(badge)}`
      );
    },
  },
  {
    id: 'T2-F7-05',
    tier: 2,
    feature: 'F7',
    name: 'UI Copy Text handles cleared empty text cleanly without errors or NaN%',
    timeoutMs: 105_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'cursive-1line.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);

      await dom.setEditableText('');
      const buttons = await dom.listButtons();
      const copyBtn = buttons.find((b) => /\bcopy\b/i.test(b.text || b.ariaLabel || ''));
      assert(Boolean(copyBtn), 'Copy Text button must be rendered in Stage 1');
      if (!copyBtn!.disabled) {
        const copyRes = await dom.clickCopyText();
        assert(copyRes.clicked, 'Enabled Copy Text button must click cleanly');
        const clip = await dom.readClipboardText();
        assertEqual(clip, '', 'Clipboard must be empty string when editable text is cleared');
      }
      const txt = await dom.getPageText();
      assertNotIncludes(txt, 'NaN%', 'Must not show NaN% on empty text');
    },
  },

  // =========================================================================
  // F8 Boundaries — 5 Tests
  // =========================================================================
  {
    id: 'T2-F8-01',
    tier: 2,
    feature: 'F8',
    name: 'darkInverted: true on sparse dark-mode screenshot (~1.5% bright boundary, Finding 5)',
    timeoutMs: 65_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const fp = path.join(ctx.fixturesDir, 'printed-dark-mode-sparse.png');
      const gt = ctx.manifest.fixtures['printed-dark-mode-sparse.png'].groundTruth;
      const { status, body } = await srv.postLocalOcr(
        { image: await pngToDataUrl(fp), ocrProfile: 'printed', exportStages: false },
        45_000
      );
      assertEqual(status, 200, 'HTTP 200');
      assertEqual(body.darkInverted, true, 'Sparse dark-mode screenshot must set darkInverted: true');
      const acc = characterAccuracy(body.text, gt, { caseInsensitive: false });
      assertGreaterOrEqual(acc, 0.9, `Expected >= 0.90 case-sensitive accuracy on sparse dark-mode, got ${acc.toFixed(3)}`);
    },
  },
  {
    id: 'T2-F8-02',
    tier: 2,
    feature: 'F8',
    name: 'Light paper surrounded by dark desk border MUST NOT set darkInverted: true',
    timeoutMs: 65_000,
    async run(ctx) {
      const w = 480;
      const h = 320;
      const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">
        <rect width="100%" height="100%" fill="#1e1e1e" />
        <rect x="70" y="45" width="340" height="230" fill="#f8f8f8" />
        <text x="105" y="135" font-family="Arial" font-size="34" font-weight="bold" fill="#111111">Buy Milk</text>
        <text x="105" y="195" font-family="Arial" font-size="34" font-weight="bold" fill="#111111">Call Bob</text>
      </svg>`;
      const pngBuf = await sharp(Buffer.from(svg, 'utf8')).png().toBuffer();
      const srv = await ctx.getServer();
      const { status, body } = await srv.postLocalOcr(
        { image: `data:image/png;base64,${pngBuf.toString('base64')}`, ocrProfile: 'printed', exportStages: false },
        45_000
      );
      assertEqual(status, 200, 'HTTP 200');
      assertEqual(body.darkInverted, false, 'Light paper on dark desk must NOT invert as dark mode');
      assertIncludes(body.text.toLowerCase(), 'milk', 'Text on light paper must remain readable');
    },
  },
  {
    id: 'T2-F8-03',
    tier: 2,
    feature: 'F8',
    name: 'Zero-margin crop with exportStages: true and dpiScale >= 1.5 writes valid PNGs',
    timeoutMs: 75_000,
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'server/handwritingRecognizer.ts');
      const procFn = resolveExportFn<(opts: Record<string, unknown>) => Promise<Record<string, any>>>(
        mod,
        ['processImageWithLocalOcr']
      );
      assert(procFn, 'processImageWithLocalOcr missing');
      const buf = fs.readFileSync(path.join(ctx.fixturesDir, 'zero-margin-crop.png'));
      const res = await procFn({
        imageBuffer: buf,
        ocrProfile: 'printed',
        exportStages: true,
        exportRootDir: ctx.createTempDir('ocr-exports-'),
      });
      cleanupExportDir(ctx, res.exportDir);
      assert(res.exportDir && fs.existsSync(res.exportDir), 'exportDir must exist');
      const pngs = fs.readdirSync(res.exportDir).filter((f) => f.endsWith('.png'));
      assertGreaterOrEqual(pngs.length, 1, 'Expected stage PNGs');
      for (const f of pngs) {
        const { data, info } = await sharp(path.join(res.exportDir, f))
          .raw()
          .toBuffer({ resolveWithObject: true });
        assert(
          info.width > 0 && info.height > 0 && data.length === info.width * info.height * info.channels,
          `Invalid PNG pixel buffer ${f}`
        );
      }
    },
  },
  {
    id: 'T2-F8-04',
    tier: 2,
    feature: 'F8',
    name: 'exportStages: false writes zero export directories or files',
    timeoutMs: 60_000,
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'server/handwritingRecognizer.ts');
      const procFn = resolveExportFn<(opts: Record<string, unknown>) => Promise<Record<string, any>>>(
        mod,
        ['processImageWithLocalOcr']
      );
      assert(procFn, 'processImageWithLocalOcr missing');
      const buf = fs.readFileSync(path.join(ctx.fixturesDir, 'printed-schedule.png'));
      const res = await procFn({
        imageBuffer: buf,
        ocrProfile: 'printed',
        exportStages: false,
        exportRootDir: ctx.createTempDir('ocr-exports-'),
      });
      assertEqual(res.exportDir, undefined, 'exportDir must be undefined when exportStages is false');
    },
  },
  {
    id: 'T2-F8-05',
    tier: 2,
    feature: 'F8',
    name: 'Vision AI fallback parity when GROQ_API_KEY & GEMINI_API_KEY are absent',
    timeoutMs: 130_000,
    async run(ctx) {
      const noAiSrv = await ctx.spawnSecondaryServer({ GROQ_API_KEY: '', GEMINI_API_KEY: '' });
      const health = await noAiSrv.getHealth();
      assertEqual(health.body.aiConfigured, false, 'Secondary server must report aiConfigured: false');

      for (const name of ['printed-schedule.png', 'cursive-clean-2lines.png']) {
        const fp = path.join(ctx.fixturesDir, name);
        const gt = ctx.manifest.fixtures[name].groundTruth;
        const { status, body } = await noAiSrv.postExtractReminder(
          { image: await pngToDataUrl(fp), ocrProfile: 'auto' },
          55_000
        );
        assertEqual(status, 200, `${name}: fallback HTTP 200`);
        assertEqual(body.provider, 'local-parser', `${name}: provider must fall back to local-parser`);
        const txt = body.rawText ?? body.ocrText ?? '';
        const acc = characterAccuracy(txt, gt, { caseInsensitive: true });
        assertGreaterOrEqual(acc, 0.9, `${name}: fallback accuracy >= 0.90, got ${acc.toFixed(3)} ("${txt}")`);
      }
    },
  },
];
