import fs from 'node:fs';
import path from 'node:path';
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

export const tier4Tests: TestCase[] = [
  {
    id: 'T4-S1',
    tier: 4,
    feature: 'SCENARIO',
    name: 'Real-world 14-line handwritten grocery note (Offline Local UI + Vision AI >= 90%)',
    timeoutMs: 180_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      const fp = path.join(ctx.fixturesDir, 'real-grocery-note.png');
      const gtRaw = ctx.manifest.fixtures['real-grocery-note.png'].groundTruth;
      const gtNorm = gtRaw.replace(/browlli/gi, 'broccoli');

      // 1. Offline Local via real headless Chrome UI
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(fp);
      await dom.waitForScanComplete(75_000, ctx.signal);

      const uiText = await dom.getEditableText();
      const uiLower = uiText.toLowerCase();
      for (const noise of ['1953', '1955', ': bread:', 'milk 5']) {
        assertNotIncludes(uiLower, noise, `Offline UI output leaked spiral/bleed noise: ${noise}`);
      }
      for (const over of ['common', 'teacher']) {
        assertNotIncludes(uiLower, over, `Offline UI over-corrected anchor word into ${over}`);
      }
      assertIncludes(uiLower, 'tomorrow', 'Offline UI preserved tomorrow');
      assertIncludes(uiLower, 'tomatoes', 'Offline UI preserved tomatoes');
      assertIncludes(uiLower, 'turkey', 'Offline UI preserved turkey');

      const offlineAcc = Math.max(
        characterAccuracy(uiText, gtRaw, { caseInsensitive: true }),
        characterAccuracy(uiText, gtNorm, { caseInsensitive: true })
      );
      assertGreaterOrEqual(
        offlineAcc,
        0.9,
        `Real grocery note Offline Local accuracy must be >= 0.90, got ${offlineAcc.toFixed(3)}\nActual:\n${uiText}`
      );

      const accBadge = await dom.getAccuracyBadge();
      assert(accBadge.found && accBadge.percent !== null, 'UI accuracy % badge missing');
      const styleBadge = await dom.getDetectedStyleBadge();
      assertEqual(styleBadge.style, 'handwriting', 'UI detected style must be handwriting');
      const lineRows = await dom.getLineConfidences();
      assertGreaterOrEqual(lineRows.length, 10, `Expected >= 10 per-line confidence rows, got ${lineRows.length}`);

      const copy = await dom.clickCopyText();
      assert(copy.clicked, 'Copy Text button missing');
      assertEqual(await dom.readClipboardText(), uiText, 'Clipboard must match extracted grocery note');

      // 2. Vision AI / online mode parity
      const vision = await srv.postExtractReminder(
        { image: await pngToDataUrl(fp), ocrProfile: 'handwriting' },
        55_000
      );
      assertEqual(vision.status, 200, 'Vision AI HTTP 200');
      const vText = vision.body.rawText ?? vision.body.ocrText ?? '';
      const visionAcc = Math.max(
        characterAccuracy(vText, gtRaw, { caseInsensitive: true }),
        characterAccuracy(vText, gtNorm, { caseInsensitive: true })
      );
      assertGreaterOrEqual(
        visionAcc,
        0.9,
        `Real grocery note Vision AI accuracy must be >= 0.90, got ${visionAcc.toFixed(3)}\nActual:\n${vText}`
      );
    },
  },
  {
    id: 'T4-S2',
    tier: 4,
    feature: 'SCENARIO',
    name: 'Slanted +20 deg handwritten meeting note on lined paper (proper nouns, dates, prices)',
    timeoutMs: 165_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      const fp = path.join(ctx.fixturesDir, 'meeting-note-slanted.png');
      const gt = ctx.manifest.fixtures['meeting-note-slanted.png'].groundTruth;

      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.clickModeButton(/^Cursive\s*\/\s*Handwriting$/i);
      await dom.uploadFixture(fp);
      await dom.waitForScanComplete(60_000, ctx.signal);

      const uiText = await dom.getEditableText();
      assertIncludes(uiText, 'Alice', 'Proper noun Alice preserved');
      assertIncludes(uiText, 'Sarah', 'Proper noun Sarah preserved');
      assertIncludes(uiText, '$450.00', 'Currency $450.00 preserved');
      assertIncludes(uiText, '3:30pm', 'Time 3:30pm preserved');
      const offlineAcc = characterAccuracy(uiText, gt, { caseInsensitive: true });
      assertGreaterOrEqual(offlineAcc, 0.9, `Offline meeting note >= 0.90, got ${offlineAcc.toFixed(3)}\n${uiText}`);

      const vision = await srv.postExtractReminder(
        { image: await pngToDataUrl(fp), ocrProfile: 'handwriting' },
        55_000
      );
      assertEqual(vision.status, 200, 'Vision AI HTTP 200');
      const vText = vision.body.rawText ?? vision.body.ocrText ?? '';
      assertGreaterOrEqual(
        characterAccuracy(vText, gt, { caseInsensitive: true }),
        0.9,
        `Vision meeting note >= 0.90, got\n${vText}`
      );
    },
  },
  {
    id: 'T4-S3',
    tier: 4,
    feature: 'SCENARIO',
    name: 'Dark-mode printed electric utility invoice -> reminder Bills/high urgency',
    timeoutMs: 150_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      const fp = path.join(ctx.fixturesDir, 'invoice-dark-mode.png');
      const gt = ctx.manifest.fixtures['invoice-dark-mode.png'].groundTruth;

      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(fp);
      await dom.waitForScanComplete(55_000, ctx.signal);

      const uiText = await dom.getEditableText();
      assertIncludes(uiText, '#8841', 'Invoice #8841 preserved');
      assertIncludes(uiText, '$114.50', 'Amount $114.50 preserved');
      assertIncludes(uiText, '12/15/2026', 'Date 12/15/2026 preserved');
      assertEqual((await dom.getDetectedStyleBadge()).style, 'printed', 'Invoice detected style printed');
      assertGreaterOrEqual(
        characterAccuracy(uiText, gt, { caseInsensitive: false }),
        0.9,
        `Invoice case-sensitive Offline >= 0.90, got "${uiText}"`
      );

      const vision = await srv.postExtractReminder({ image: await pngToDataUrl(fp), ocrProfile: 'printed' }, 45_000);
      assertEqual(vision.status, 200, 'Vision HTTP 200');
      const visionText = vision.body.rawText ?? vision.body.ocrText ?? '';
      assertGreaterOrEqual(
        characterAccuracy(visionText, gt, { caseInsensitive: false }),
        0.9,
        `Invoice case-sensitive Vision >= 0.90, got "${visionText}"`
      );
      assertEqual(vision.body.data?.category, 'Bills', 'Category must be Bills');
      assertEqual(vision.body.data?.priority, 'high', 'Priority must be high for urgent bill');
    },
  },
  {
    id: 'T4-S4',
    tier: 4,
    feature: 'SCENARIO',
    name: 'Mixed printed header + cursive handwritten body document (Auto mode)',
    timeoutMs: 160_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      const fp = path.join(ctx.fixturesDir, 'mixed-printed-cursive.png');
      const gt = ctx.manifest.fixtures['mixed-printed-cursive.png'].groundTruth;

      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(fp);
      await dom.waitForScanComplete(60_000, ctx.signal);

      const uiText = await dom.getEditableText();
      assertEqual((await dom.getDetectedStyleBadge()).style, 'mixed', 'UI style badge must show mixed');
      assertIncludes(uiText.toUpperCase(), 'WEEKLY ERRANDS', 'Printed header preserved');
      assertIncludes(uiText.toLowerCase(), 'salmon', 'Cursive body salmon preserved');
      assertGreaterOrEqual(characterAccuracy(uiText, gt, { caseInsensitive: true }), 0.9, 'Mixed Offline >= 0.90');

      const vision = await srv.postExtractReminder({ image: await pngToDataUrl(fp), ocrProfile: 'auto' }, 50_000);
      assertEqual(vision.status, 200, 'Vision HTTP 200');
      const vText = vision.body.rawText ?? vision.body.ocrText ?? '';
      assertGreaterOrEqual(characterAccuracy(vText, gt, { caseInsensitive: true }), 0.9, 'Mixed Vision >= 0.90');
    },
  },
  {
    id: 'T4-S5',
    tier: 4,
    feature: 'SCENARIO',
    name: 'Full Snap2 scan -> OCR metadata -> Convert to Reminder -> Save Reminder persistence',
    timeoutMs: 210_000,
    async run(ctx) {
      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      const fp = path.join(ctx.fixturesDir, 'reminder-full-flow.png');
      const gt = ctx.manifest.fixtures['reminder-full-flow.png'].groundTruth;

      // Offline parity check
      const localRes = await srv.postLocalOcr(
        { image: await pngToDataUrl(fp), ocrProfile: 'handwriting', exportStages: false },
        55_000
      );
      assertEqual(localRes.status, 200, 'Local OCR HTTP 200');
      assertGreaterOrEqual(
        characterAccuracy(localRes.body.text, gt, { caseInsensitive: true }),
        0.9,
        `Local OCR >= 0.90 ("${localRes.body.text}")`
      );

      // Live browser Auto mode end-to-end flow
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.uploadFixture(fp);
      await dom.waitForScanComplete(55_000, ctx.signal);

      const uiText = await dom.getEditableText();
      assertGreaterOrEqual(
        characterAccuracy(uiText, gt, { caseInsensitive: true }),
        0.9,
        `Browser Auto OCR >= 0.90 ("${uiText}")`
      );
      const badge = await dom.getAccuracyBadge();
      assert(badge.found && badge.percent !== null, 'Accuracy badge shown');
      assertGreaterOrEqual((await dom.getLineConfidences()).length, 2, 'Line confidences shown');

      const copy = await dom.clickCopyText();
      assert(copy.clicked, 'Copy Text works');
      assertEqual(await dom.readClipboardText(), uiText, 'Clipboard matches Stage 1 text');

      await dom.clickConvertToReminder(40_000, ctx.signal);
      await dom.clickSaveReminder(15_000, ctx.signal);

      const saved = await dom.getStoredReminders();
      const found = saved.find((r) => typeof r.ocrText === 'string' && r.ocrText.toLowerCase().includes('dentist'));
      assert(Boolean(found), 'Reminder persisted in localStorage with ocrText');
    },
  },
  {
    id: 'T4-S6',
    tier: 4,
    feature: 'SCENARIO',
    name: 'Handwritten confusion checklist + exportStages PNGs + Copy Text',
    timeoutMs: 130_000,
    async run(ctx) {
      const mod = await loadOptionalModule(ctx.repoRoot, 'server/handwritingRecognizer.ts');
      const procFn = resolveExportFn<(opts: Record<string, unknown>) => Promise<Record<string, any>>>(
        mod,
        ['processImageWithLocalOcr']
      );
      assert(procFn, 'processImageWithLocalOcr missing');

      const fp = path.join(ctx.fixturesDir, 'checklist-confusions.png');
      const gt = ctx.manifest.fixtures['checklist-confusions.png'].groundTruth;
      const expRes = await procFn({
        imageBuffer: fs.readFileSync(fp),
        ocrProfile: 'handwriting',
        exportStages: true,
        exportRootDir: ctx.createTempDir('ocr-exports-'),
      });
      cleanupExportDir(ctx, expRes.exportDir);
      assert(expRes.exportDir && fs.existsSync(expRes.exportDir), 'exportDir created');

      const srv = await ctx.getServer();
      const dom = await ctx.getDom();
      await dom.resetPage(srv.baseUrl, ctx.signal);
      await dom.clickModeButton(/^Offline Local$/i);
      await dom.uploadFixture(fp);
      await dom.waitForScanComplete(55_000, ctx.signal);

      const uiText = await dom.getEditableText();
      const acc = characterAccuracy(uiText, gt, { caseInsensitive: true });
      assertGreaterOrEqual(acc, 0.9, `Checklist accuracy >= 0.90, got ${acc.toFixed(3)} ("${uiText}")`);
      for (const expectedWord of ['morning', 'milk', 'date', 'work', 'spinach', 'dinner']) {
        assertIncludes(uiText.toLowerCase(), expectedWord, `Missing checklist item ${expectedWord}`);
      }
      const copy = await dom.clickCopyText();
      assert(copy.clicked, 'Copy Text clicked');
      assertEqual(await dom.readClipboardText(), uiText, 'Clipboard matches multi-line checklist');
    },
  },
];
