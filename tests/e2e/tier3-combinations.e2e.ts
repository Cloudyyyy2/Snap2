import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { preprocessForOCR } from '../../src/services/imageProcessor';
import { binarizeSauvola } from '../../src/lib/ocr/preprocess';
import { segmentLines, cropLine } from '../../src/lib/ocr/handwriting/lines';
import {
  assert,
  assertEqual,
  assertGreaterOrEqual,
  assertIncludes,
  assertNotIncludes,
  characterAccuracy,
  loadOptionalModule,
  resolveExportFn,
} from './helpers/metrics';
import type { TestCase, TestContext } from './helpers/types';

async function pngToDataUrl(filePath: string): Promise<string> {
  return `data:image/png;base64,${fs.readFileSync(filePath).toString('base64')}`;
}

async function dataUrlToIntermediateCount(dataUrl: string): Promise<number> {
  const commaIdx = dataUrl.indexOf(',');
  const b64 = commaIdx >= 0 ? dataUrl.slice(commaIdx + 1) : dataUrl;
  const { data } = await sharp(Buffer.from(b64, 'base64')).grayscale().raw().toBuffer({ resolveWithObject: true });
  let count = 0;
  for (let i = 0; i < data.length; i++) {
    if (data[i] >= 15 && data[i] <= 240) count++;
  }
  return count;
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

export const tier3Tests: TestCase[] = [
  {
    id: 'T3-01',
    tier: 3,
    feature: 'PAIRWISE',
    name: 'F1 + F2: Anti-aliased grayscale normalization + cursive slant normalization',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/geometry.ts');
      const deslantFn = resolveExportFn<(...args: any[]) => Record<string, any>>(mod, ['deslantGrayscale']);
      assert(deslantFn, 'deslantGrayscale missing');

      const { data, info } = await sharp(path.join(ctx.fixturesDir, 'cursive-slanted-pos22.png'))
        .ensureAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });
      const pre = preprocessForOCR({ data: new Uint8ClampedArray(data), width: info.width, height: info.height });
      const gray = new Uint8ClampedArray(info.width * info.height);
      let midPre = 0;
      for (let i = 0; i < gray.length; i++) {
        const v = pre.data[i * 4];
        gray[i] = v;
        if (v >= 15 && v <= 240) midPre++;
      }
      assertGreaterOrEqual(midPre, 40, 'F1 grayscale intermediate tones preserved');

      const binary = binarizeSauvola(gray, info.width, info.height, 15, 0.2);
      const res = deslantFn(gray, binary, info.width, info.height, { minAngleDeg: -38, maxAngleDeg: 38 });
      assert(Boolean(res.applied ?? res.corrected ?? res.deslanted), 'F2 deslant applied');
      let midAfter = 0;
      for (let i = 0; i < res.data.length; i++) {
        if (res.data[i] >= 15 && res.data[i] <= 240) midAfter++;
      }
      assertGreaterOrEqual(midAfter, 40, 'Anti-aliased grays preserved after shear deslant');
    },
  },
  {
    id: 'T3-02',
    tier: 3,
    feature: 'PAIRWISE',
    name: 'F2 + F3: Slant normalization + ascender/descender multi-line segmentation',
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'src/lib/ocr/geometry.ts');
      const deslantFn = resolveExportFn<(...args: any[]) => Record<string, any>>(mod, ['deslantGrayscale']);
      assert(deslantFn, 'deslantGrayscale missing');

      const { data, info } = await sharp(path.join(ctx.fixturesDir, 'meeting-note-slanted.png'))
        .grayscale()
        .raw()
        .toBuffer({ resolveWithObject: true });
      const gray = new Uint8ClampedArray(data);
      const binary = binarizeSauvola(gray, info.width, info.height, 19, 0.2);
      const deslanted = deslantFn(gray, binary, info.width, info.height, { minAngleDeg: -38, maxAngleDeg: 38 });
      assert(Boolean(deslanted.applied ?? deslanted.corrected ?? deslanted.deslanted), 'Expected deslant on meeting note');

      const desBin = binarizeSauvola(deslanted.data, deslanted.width, deslanted.height, 19, 0.2);
      const boxes = segmentLines(desBin, deslanted.width, deslanted.height);
      assertGreaterOrEqual(boxes.length, 3, `Expected >= 3 segmented lines after deslanting, got ${boxes.length}`);
      for (const box of boxes) {
        const crop = cropLine(deslanted.data, deslanted.width, deslanted.height, box, 14);
        assertGreaterOrEqual(crop.height, 36, 'Cropped line preserves ascender/descender height');
      }
    },
  },
  {
    id: 'T3-03',
    tier: 3,
    feature: 'PAIRWISE',
    name: 'F1 + F3 + F4: Spiral/bleed cleanup + >=14px TrOCR padding + beam search',
    timeoutMs: 75_000,
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'server/handwritingRecognizer.ts');
      const procFn = resolveExportFn<(opts: Record<string, unknown>) => Promise<Record<string, any>>>(
        mod,
        ['processImageWithLocalOcr']
      );
      assert(procFn, 'processImageWithLocalOcr missing');
      const fp = path.join(ctx.fixturesDir, 'cursive-spiral-bleed.png');
      const gt = ctx.manifest.fixtures['cursive-spiral-bleed.png'].groundTruth;
      const res = await procFn({
        imageBuffer: fs.readFileSync(fp),
        ocrProfile: 'handwriting',
        exportStages: true,
        exportRootDir: ctx.createTempDir('ocr-exports-'),
      });
      cleanupExportDir(ctx, res.exportDir);

      assertEqual(res.lines.length, 3, `Expected strictly 3 text lines after spiral cleanup, got ${res.lines.length}`);
      const acc = characterAccuracy(res.text, gt, { caseInsensitive: true });
      assertGreaterOrEqual(acc, 0.9, `Expected >= 0.90 accuracy on spiral bleed fixture, got ${acc.toFixed(3)} ("${res.text}")`);
    },
  },
  {
    id: 'T3-04',
    tier: 3,
    feature: 'PAIRWISE',
    name: 'F4 + F5: TrOCR beam search + contextual post-processing preserves anchors & currency',
    timeoutMs: 75_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const fp = path.join(ctx.fixturesDir, 'cursive-confusions.png');
      const gt = ctx.manifest.fixtures['cursive-confusions.png'].groundTruth;
      const { status, body } = await srv.postLocalOcr(
        { image: await pngToDataUrl(fp), ocrProfile: 'handwriting', exportStages: false },
        55_000
      );
      assertEqual(status, 200, 'HTTP 200');
      assertIncludes(body.text, '$45.50', 'Currency preserved');
      assertIncludes(body.text, 'Alice', 'Proper noun preserved');
      const acc = characterAccuracy(body.text, gt, { caseInsensitive: true });
      assertGreaterOrEqual(acc, 0.9, `Expected >= 0.90 accuracy, got ${acc.toFixed(3)} ("${body.text}")`);
    },
  },
  {
    id: 'T3-05',
    tier: 3,
    feature: 'PAIRWISE',
    name: 'F1 + F5 + F8: Dark-mode inversion + post-processing + printed schedule accuracy',
    timeoutMs: 75_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const fp = path.join(ctx.fixturesDir, 'printed-dark-mode.png');
      const gt = ctx.manifest.fixtures['printed-dark-mode.png'].groundTruth;
      const { status, body } = await srv.postLocalOcr(
        { image: await pngToDataUrl(fp), ocrProfile: 'auto', exportStages: false },
        45_000
      );
      assertEqual(status, 200, 'HTTP 200');
      assertEqual(body.darkInverted, true, 'darkInverted: true');
      assertEqual(body.detectedStyle, 'printed', 'detectedStyle: printed');
      assertIncludes(body.text, '9:30am', 'Time preserved');
      assertIncludes(body.text, '402', 'Room number preserved');
      const acc = characterAccuracy(body.text, gt, { caseInsensitive: false });
      assertGreaterOrEqual(acc, 0.9, `Expected >= 0.90 case-sensitive accuracy, got ${acc.toFixed(3)}`);
    },
  },
  {
    id: 'T3-06',
    tier: 3,
    feature: 'PAIRWISE',
    name: 'F3 + F4 + F8: Small-font wide-margin cursive triggers dpi-upscale + auto-bbox-shrink + saveGrayscalePng',
    timeoutMs: 75_000,
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'server/handwritingRecognizer.ts');
      const procFn = resolveExportFn<(opts: Record<string, unknown>) => Promise<Record<string, any>>>(
        mod,
        ['processImageWithLocalOcr']
      );
      assert(procFn, 'processImageWithLocalOcr missing');
      const fp = path.join(ctx.fixturesDir, 'cursive-wide-margin.png');
      const gt = ctx.manifest.fixtures['cursive-wide-margin.png'].groundTruth;
      const res = await procFn({
        imageBuffer: fs.readFileSync(fp),
        ocrProfile: 'handwriting',
        exportStages: true,
        exportRootDir: ctx.createTempDir('ocr-exports-'),
      });
      cleanupExportDir(ctx, res.exportDir);

      const steps: string[] = res.appliedSteps ?? [];
      assert(steps.some((s) => s.startsWith('dpi-upscale(')), 'dpi-upscale applied');
      assert(steps.some((s) => s.startsWith('auto-bbox-shrink(')), 'auto-bbox-shrink applied');
      assert(res.exportDir && fs.existsSync(res.exportDir), 'Export dir exists cleanly');
      const acc = characterAccuracy(res.text, gt, { caseInsensitive: true });
      assertGreaterOrEqual(acc, 0.9, `Expected >= 0.90 accuracy after upscale+shrink, got ${acc.toFixed(3)} ("${res.text}")`);
    },
  },
  {
    id: 'T3-07',
    tier: 3,
    feature: 'PAIRWISE',
    name: 'F6 + F1 + F7: UI Offline Local mode -> grayscale payload -> metadata + Copy Text',
    timeoutMs: 105_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.enableNetworkCapture();
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'cursive-clean-2lines.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);

      const reqs = dom.getCapturedRequests().filter((r) => r.url.includes('/api/local-ocr') && r.method === 'POST');
      assertGreaterOrEqual(reqs.length, 1, 'Routed to /api/local-ocr');
      const imgUrl = String(reqs[0].parsedBody?.image ?? '');
      assertGreaterOrEqual(await dataUrlToIntermediateCount(imgUrl), 40, 'Grayscale stroke tones preserved in upload');

      const accBadge = await dom.getAccuracyBadge();
      assert(accBadge.found && accBadge.percent !== null, 'Accuracy badge rendered');
      const styleBadge = await dom.getDetectedStyleBadge();
      assertEqual(styleBadge.style, 'handwriting', 'Style badge rendered as handwriting');
      const lines = await dom.getLineConfidences();
      assertGreaterOrEqual(lines.length, 2, 'Line confidences rendered');

      const copy = await dom.clickCopyText();
      assert(copy.clicked, 'Copy Text clicked');
      assertEqual(await dom.readClipboardText(), await dom.getEditableText(), 'Clipboard matches editable text');
    },
  },
  {
    id: 'T3-08',
    tier: 3,
    feature: 'PAIRWISE',
    name: 'F6 + F7: Sequential in-session mode switch cycle updates routing & UI metadata without reload',
    timeoutMs: 220_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.enableNetworkCapture();

      // 1. Printed mode in Offline Local
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.clickModeButton(/^Printed$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'printed-schedule.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);
      const reqs1 = dom.getCapturedRequests().filter((r) => r.method === 'POST' && r.url.includes('/api/local-ocr'));
      assertGreaterOrEqual(reqs1.length, 1, 'Step 1 called /api/local-ocr');
      assertEqual(reqs1[0].parsedBody?.ocrProfile, 'printed', 'Step 1 sent printed profile');
      assertEqual((await dom.getDetectedStyleBadge()).style, 'printed', 'Step 1 style printed');

      // Close modal WITHOUT page reload and switch mode in-session
      if (await dom.hasModeButton(/Close review sheet|^Close$/i)) {
        await dom.clickModeButton(/Close review sheet|^Close$/i);
      }
      dom.clearNetworkCapture();

      // 2. Switch to Cursive/Handwriting in-session
      await dom.clickModeButton(/^Cursive\s*\/\s*Handwriting$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'cursive-1line.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);
      const reqs2 = dom.getCapturedRequests().filter((r) => r.method === 'POST' && r.url.includes('/api/'));
      assertGreaterOrEqual(reqs2.length, 1, 'Step 2 dispatched OCR request');
      assertEqual(reqs2[0].parsedBody?.ocrProfile, 'handwriting', 'Step 2 sent handwriting profile');
      assertEqual((await dom.getDetectedStyleBadge()).style, 'handwriting', 'Step 2 style handwriting');

      // Close modal WITHOUT page reload and switch to Auto in-session
      if (await dom.hasModeButton(/Close review sheet|^Close$/i)) {
        await dom.clickModeButton(/Close review sheet|^Close$/i);
      }
      dom.clearNetworkCapture();

      // 3. Switch to Auto in-session
      await dom.clickModeButton(/^Auto$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'printed-schedule.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);
      const reqs3 = dom.getCapturedRequests().filter((r) => r.method === 'POST' && r.url.includes('/api/'));
      assertGreaterOrEqual(reqs3.length, 1, 'Step 3 dispatched OCR request');
      assertEqual(reqs3[0].parsedBody?.ocrProfile, 'auto', 'Step 3 sent auto profile');
      assertEqual((await dom.getDetectedStyleBadge()).style, 'printed', 'Step 3 style printed');
    },
  },
  {
    id: 'T3-09',
    tier: 3,
    feature: 'PAIRWISE',
    name: 'F6 + F5 + F8: Vision AI (/api/extract-reminder) vs Offline Local (/api/local-ocr) parity',
    timeoutMs: 125_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const fp = path.join(ctx.fixturesDir, 'cursive-confusions.png');
      const gt = ctx.manifest.fixtures['cursive-confusions.png'].groundTruth;
      const image = await pngToDataUrl(fp);

      const localRes = await srv.postLocalOcr({ image, ocrProfile: 'handwriting', exportStages: false }, 55_000);
      assertEqual(localRes.status, 200, 'Local HTTP 200');
      assertGreaterOrEqual(
        characterAccuracy(localRes.body.text, gt, { caseInsensitive: true }),
        0.9,
        'Local OCR >= 0.90'
      );

      const visionRes = await srv.postExtractReminder({ image, ocrProfile: 'handwriting' }, 55_000);
      assertEqual(visionRes.status, 200, 'Vision HTTP 200');
      const vText = visionRes.body.rawText ?? visionRes.body.ocrText ?? '';
      assert(Array.isArray(visionRes.body.lines) && visionRes.body.lines.length >= 1, 'Vision returns lines[]');
      assert(typeof visionRes.body.accuracyScore === 'number', 'Vision returns accuracyScore');
      assertGreaterOrEqual(
        characterAccuracy(vText, gt, { caseInsensitive: true }),
        0.9,
        `Vision OCR >= 0.90, got "${vText}"`
      );
    },
  },
  {
    id: 'T3-10',
    tier: 3,
    feature: 'PAIRWISE',
    name: 'F3 + F4 + F6: Mixed printed + cursive document under Auto vs Printed vs Handwriting',
    timeoutMs: 225_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const fp = path.join(ctx.fixturesDir, 'mixed-printed-cursive.png');
      const gt = ctx.manifest.fixtures['mixed-printed-cursive.png'].groundTruth;
      const image = await pngToDataUrl(fp);

      const autoRes = await srv.postLocalOcr({ image, ocrProfile: 'auto', exportStages: false }, 75_000);
      assertEqual(autoRes.status, 200, 'Auto HTTP 200');
      assertEqual(autoRes.body.detectedStyle, 'mixed', `Expected mixed style, got ${autoRes.body.detectedStyle}`);
      assertGreaterOrEqual(
        characterAccuracy(autoRes.body.text, gt, { caseInsensitive: true }),
        0.9,
        `Auto mixed >= 0.90, got "${autoRes.body.text}"`
      );

      const prRes = await srv.postLocalOcr({ image, ocrProfile: 'printed', exportStages: false }, 60_000);
      assertEqual(prRes.status, 200, 'Printed HTTP 200');
      assertEqual(prRes.body.success, true, 'Printed success true');
      assertEqual(prRes.body.recognizer, 'tesseract-host', 'Printed profile uses tesseract-host');
      assertGreaterOrEqual(prRes.body.text.trim().length, 10, 'Printed profile must return non-empty recognized text');
      assertIncludes(prRes.body.text.toUpperCase(), 'WEEKLY ERRANDS', 'Printed profile must read printed header');

      const hwRes = await srv.postLocalOcr({ image, ocrProfile: 'handwriting', exportStages: false }, 75_000);
      assertEqual(hwRes.status, 200, 'Handwriting HTTP 200');
      assertEqual(hwRes.body.success, true, 'Handwriting success true');
      assertEqual(hwRes.body.recognizer, 'trocr-host', 'Handwriting profile uses trocr-host');
      assertGreaterOrEqual(hwRes.body.text.trim().length, 10, 'Handwriting profile must return non-empty recognized text');
      assertIncludes(hwRes.body.text.toLowerCase(), 'salmon', 'Handwriting profile must read cursive body');
    },
  },
  {
    id: 'T3-11',
    tier: 3,
    feature: 'PAIRWISE',
    name: 'F4 + F5 + F7: Grocery anchor subset in UI renders per-line confidence with zero overcorrection',
    timeoutMs: 105_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.clickModeButton(/^Cursive\s*\/\s*Handwriting$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'cursive-anchors-grocery.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);

      const txt = (await dom.getEditableText()).toLowerCase();
      assertIncludes(txt, 'tomorrow', 'tomorrow preserved');
      assertIncludes(txt, 'tomatoes', 'tomatoes preserved');
      assertIncludes(txt, 'turkey', 'turkey preserved');
      assertNotIncludes(txt, 'common', 'No tomorrow->common');
      assertNotIncludes(txt, 'teacher', 'No tomatoes->teacher');
      assertNotIncludes(txt, 'they', 'No turkey->they');

      const lines = await dom.getLineConfidences();
      assertGreaterOrEqual(lines.length, 4, 'Per-line confidence rows displayed');
    },
  },
  {
    id: 'T3-12',
    tier: 3,
    feature: 'PAIRWISE',
    name: 'F1 + F6 + F9: UI Auto mode -> Stage 1 edit -> Stage 2 Convert to Reminder sends ocrText',
    timeoutMs: 145_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.enableNetworkCapture();
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'reminder-full-flow.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);

      const original = await dom.getEditableText();
      const edited = `${original}\nBring photo ID`;
      await dom.setEditableText(edited);

      dom.clearNetworkCapture();
      await dom.clickConvertToReminder(40_000, ctx.signal);

      // If AI enhancement toggle was off in Stage 1, trigger Stage 2 "Create List with AI" if present
      if (!dom.getCapturedRequests().some((r) => r.url.includes('/api/extract-reminder'))) {
        if (await dom.hasModeButton(/Create List with AI/i)) {
          await dom.clickModeButton(/Create List with AI/i);
          await new Promise((r) => setTimeout(r, 1500));
        }
      }

      const stage2Reqs = dom.getCapturedRequests().filter((r) => r.url.includes('/api/extract-reminder'));
      assertGreaterOrEqual(stage2Reqs.length, 1, 'Stage 2 called /api/extract-reminder');
      assertEqual(stage2Reqs[0].parsedBody?.ocrText, edited, 'Stage 2 sent edited ocrText');
      assertEqual(stage2Reqs[0].parsedBody?.image, undefined, 'Stage 2 should not re-upload raw image');
    },
  },
  {
    id: 'T3-13',
    tier: 3,
    feature: 'PAIRWISE',
    name: 'F2 + F4 + F5: Slanted +22 deg cursive note -> deslant -> TrOCR beam search -> postproc >= 90%',
    timeoutMs: 75_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const fp = path.join(ctx.fixturesDir, 'cursive-slanted-pos22.png');
      const gt = ctx.manifest.fixtures['cursive-slanted-pos22.png'].groundTruth;
      const { status, body } = await srv.postLocalOcr(
        { image: await pngToDataUrl(fp), ocrProfile: 'handwriting', exportStages: false },
        55_000
      );
      assertEqual(status, 200, 'HTTP 200');
      const acc = characterAccuracy(body.text, gt, { caseInsensitive: true });
      assertGreaterOrEqual(acc, 0.9, `Expected >= 0.90 accuracy on +22° slanted note, got ${acc.toFixed(3)} ("${body.text}")`);
    },
  },
  {
    id: 'T3-14',
    tier: 3,
    feature: 'PAIRWISE',
    name: 'F7 + F9: UI Copy Text + Save Reminder persists reminder in localStorage',
    timeoutMs: 160_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(path.join(ctx.fixturesDir, 'reminder-full-flow.png'));
      await dom.waitForScanComplete(55_000, ctx.signal);

      const text = await dom.getEditableText();
      const copy = await dom.clickCopyText();
      assert(copy.clicked, 'Copy Text works');
      assertEqual(await dom.readClipboardText(), text, 'Clipboard populated');

      await dom.clickConvertToReminder(40_000, ctx.signal);
      await dom.clickSaveReminder(15_000, ctx.signal);

      const saved = await dom.getStoredReminders();
      const match = saved.find((r) => typeof r.ocrText === 'string' && r.ocrText.includes('insurance'));
      assert(Boolean(match), 'Saved reminder with ocrText must persist in localStorage');
    },
  },
];
