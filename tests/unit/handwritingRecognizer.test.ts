import fs from 'fs';
import os from 'os';
import path from 'path';
import sharp from 'sharp';
import { describe, it, expect } from 'vitest';
import {
  isHandwritingModelAvailable,
  getActiveHandwritingModelName,
  processImageWithLocalOcr,
} from '../../server/handwritingRecognizer';
import { parseReminderLocally } from '../../server/localReminderParser';

import {
  isPrintedModelAvailable,
  getActivePrintedModelName,
} from '../../server/printedRecognizer';

describe('local dual offline OCR recognizers (TrOCR Base + tessdata_full) & offline parser', () => {
  it('detects downloaded local handwriting and printed models and rejects missing dirs', () => {
    const hasHandwriting = isHandwritingModelAvailable();
    const hasPrinted = isPrintedModelAvailable();

    if (hasHandwriting) {
      expect(getActiveHandwritingModelName()).toBe('trocr-base-handwritten');
    } else {
      expect(getActiveHandwritingModelName()).toBe('missing');
    }
    expect(isHandwritingModelAvailable('/tmp/nonexistent-trocr-model-dir')).toBe(false);

    if (hasPrinted) {
      expect(getActivePrintedModelName()).toBe('tessdata_full');
    } else {
      expect(getActivePrintedModelName()).toBe('missing');
    }
    expect(isPrintedModelAvailable('/tmp/nonexistent-tessdata-dir')).toBe(false);
  });

  it(
    'runs TrOCR offline for handwriting and Tesseract offline for printed (including dark-mode soft-copy)',
    async () => {
      const svg = `
        <svg width="460" height="160" xmlns="http://www.w3.org/2000/svg">
          <rect width="460" height="160" fill="white" />
          <line x1="0" y1="68" x2="460" y2="68" stroke="#c8d0dc" stroke-width="1.5" />
          <line x1="0" y1="128" x2="460" y2="128" stroke="#c8d0dc" stroke-width="1.5" />
          <text x="24" y="60" font-size="34" font-family="sans-serif" fill="#18181b">buy milk</text>
          <text x="24" y="120" font-size="34" font-family="sans-serif" fill="#18181b">call dentist</text>
        </svg>
      `;
      if (!isHandwritingModelAvailable()) {
        console.log('[test skip] Local TrOCR weights not present on disk - skipping inference test');
        return;
      }

      const pngBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
      const tempExportRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'snapremind-ocr-test-'));

      try {
        const res = await processImageWithLocalOcr({
          imageBuffer: pngBuffer,
          ocrProfile: 'handwriting',
          exportStages: true,
          exportRootDir: tempExportRoot,
        });

        expect(res.recognizer).toBe('trocr-host');
        expect(res.lines.length).toBe(2);
        expect(res.text.trim().length).toBeGreaterThan(0);
        expect(res.exportDir).toBeDefined();
        expect(res.appliedSteps).toContain('grayscale-luminance');
        expect(res.appliedSteps).toContain('median-3x3');
        expect(res.processedPreview?.startsWith('data:image/png;base64,')).toBe(true);

        const dir = res.exportDir!;
        expect(fs.existsSync(path.join(dir, '03b_background_flattened.png'))).toBe(true);
        expect(fs.existsSync(path.join(dir, '03c_ruled_lines_removed.png'))).toBe(true);
        expect(fs.existsSync(path.join(dir, '04_contrast_enhanced.png'))).toBe(true);
        expect(fs.existsSync(path.join(dir, '05_adaptive_binary.png'))).toBe(true);
        expect(fs.existsSync(path.join(dir, '06_line_01.png'))).toBe(true);
        expect(fs.existsSync(path.join(dir, '06_line_02.png'))).toBe(true);

        const report = fs.readFileSync(path.join(dir, 'extracted_text.txt'), 'utf8');
        expect(report).toContain('Engine: TrOCR');

        // Confirm printed mode routes to offline Tesseract LSTM without touching network
        const printedRes = await processImageWithLocalOcr({
          imageBuffer: pngBuffer,
          ocrProfile: 'printed',
          exportStages: false,
        });
        expect(printedRes.recognizer).toBe('tesseract-host');
        expect(printedRes.text.toLowerCase()).toContain('buy milk');
        expect(printedRes.text.toLowerCase()).toContain('call dentist');

        // Confirm dark-mode soft-copy printed UI screenshot gets inverted & recognized cleanly offline
        const darkSvg = `
          <svg width="520" height="180" xmlns="http://www.w3.org/2000/svg">
            <rect width="520" height="180" fill="#111318" />
            <text x="36" y="72" font-size="34" font-family="sans-serif" fill="#f4f4f5">Pay water bill</text>
            <text x="36" y="132" font-size="34" font-family="sans-serif" fill="#f4f4f5">Due Friday 5pm</text>
          </svg>
        `;
        const darkPngBuffer = await sharp(Buffer.from(darkSvg)).png().toBuffer();
        const darkRes = await processImageWithLocalOcr({
          imageBuffer: darkPngBuffer,
          ocrProfile: 'printed',
          exportStages: false,
        });
        expect(darkRes.darkInverted).toBe(true);
        expect(darkRes.recognizer).toBe('tesseract-host');
        expect(darkRes.text.toLowerCase()).toContain('pay water bill');
        expect(darkRes.text.toLowerCase()).toContain('friday');
      } finally {
        fs.rmSync(tempExportRoot, { recursive: true, force: true });
      }
    },
    45000
  );

  it('parseReminderLocally builds structured reminders from handwritten lines', () => {
    const parsed = parseReminderLocally(
      'Pay electric bill $114 tomorrow 3pm\nCheck meter reading\nUrgent',
      undefined,
      ['handwritten']
    );

    expect(parsed.title).toContain('Pay electric bill');
    expect(parsed.category).toBe('Bills');
    expect(parsed.priority).toBe('high');
    expect(parsed.checklist.length).toBe(3);
    expect(parsed.tags).toContain('handwritten');
    expect(new Date(parsed.dueDate).getHours()).toBe(15);
  });

  it(
    'handles blank/empty frames safely and auto-detects mixed printed + handwritten regions offline',
    async () => {
      // 1. Blank white image edge case
      const blankSvg = `<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg"><rect width="400" height="200" fill="white" /></svg>`;
      const blankPng = await sharp(Buffer.from(blankSvg)).png().toBuffer();
      const blankRes = await processImageWithLocalOcr({
        imageBuffer: blankPng,
        ocrProfile: 'auto',
        exportStages: false,
      });
      expect(blankRes.text).toBe('');
      expect(blankRes.lines.length).toBe(0);
      expect(blankRes.qualityScore).toBe(0);
      expect(blankRes.appliedSteps).toContain('no-ink-detected');

      // 2. Mixed printed header region + connected cursive-like handwriting path region separated vertically
      if (!isPrintedModelAvailable() && !isHandwritingModelAvailable()) {
        console.log('[test skip] Offline OCR models not downloaded - skipping mixed OCR assertions');
        return;
      }

      const mixedSvg = `
        <svg width="640" height="360" xmlns="http://www.w3.org/2000/svg">
          <rect width="640" height="360" fill="white" />
          <text x="48" y="68" font-size="34" font-family="sans-serif" fill="#111827">PAY ELECTRIC BILL</text>
          <path d="M52 248 C70 212, 92 260, 114 226 C132 204, 152 256, 178 228 C196 212, 224 252, 248 222 C270 202, 292 254, 326 226 C348 210, 376 250, 412 224" fill="none" stroke="#18181b" stroke-width="5" stroke-linecap="round" />
          <path d="M446 232 C464 214, 488 250, 518 222 C540 204, 560 250, 588 228" fill="none" stroke="#18181b" stroke-width="4.5" stroke-linecap="round" />
        </svg>
      `;
      const mixedPng = await sharp(Buffer.from(mixedSvg)).png().toBuffer();
      const mixedRes = await processImageWithLocalOcr({
        imageBuffer: mixedPng,
        ocrProfile: 'auto',
        exportStages: false,
      });

      expect(mixedRes.regionsCount).toBeGreaterThanOrEqual(2);
      expect(mixedRes.text.toLowerCase()).toContain('pay electric bill');
      expect(['mixed', 'printed', 'handwriting']).toContain(mixedRes.detectedStyle);
      if (mixedRes.detectedStyle === 'mixed') {
        expect(mixedRes.recognizer).toBe('hybrid-host');
      }
    },
    45000
  );

  it(
    'preserves and recognizes text from a high-res shaded mobile camera photo with dark desk borders and vertical margin rule',
    async () => {
      if (!isPrintedModelAvailable() && !isHandwritingModelAvailable()) {
        console.log('[test skip] Offline OCR models not downloaded - skipping shaded phone camera test');
        return;
      }

      const phoneSvg = `
        <svg width="1080" height="1440" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="indoorShade" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#dad4c8" />
              <stop offset="60%" stop-color="#c4bdb0" />
              <stop offset="100%" stop-color="#a8a092" />
            </linearGradient>
          </defs>
          <!-- Dark wooden table background -->
          <rect width="1080" height="1440" fill="#2b2119" />
          <!-- Tilted / slightly inset shaded notebook page -->
          <polygon points="85,70 1010,115 965,1370 55,1320" fill="url(#indoorShade)" />
          <!-- Notebook vertical margin line & horizontal ruled lines -->
          <line x1="210" y1="90" x2="190" y2="1320" stroke="#9e5a5a" stroke-width="4" />
          <line x1="90" y1="360" x2="970" y2="380" stroke="#8c96a4" stroke-width="2" />
          <line x1="90" y1="490" x2="970" y2="510" stroke="#8c96a4" stroke-width="2" />
          <!-- Bold handwritten-style / printed note lines -->
          <text x="250" y="345" font-size="68" font-family="sans-serif" font-weight="600" fill="#1b1f26">Buy groceries tomorrow</text>
          <text x="250" y="475" font-size="68" font-family="sans-serif" font-weight="600" fill="#1b1f26">Call dentist at 3pm</text>
        </svg>
      `;
      const phonePng = await sharp(Buffer.from(phoneSvg)).jpeg({ quality: 88 }).toBuffer();
      const res = await processImageWithLocalOcr({
        imageBuffer: phonePng,
        ocrProfile: 'auto',
        exportStages: false,
      });

      expect(res.lines.length).toBeGreaterThanOrEqual(1);
      expect(res.text.toLowerCase()).toContain('groceries');
      expect(res.text.toLowerCase()).toContain('dentist');
      expect(res.appliedSteps).not.toContain('no-ink-detected');
    },
    45000
  );

  it('saveGrayscalePng succeeds after auto-bbox-shrink without memory area too small error', async () => {
    const svg = `
      <svg width="460" height="160" xmlns="http://www.w3.org/2000/svg">
        <rect width="460" height="160" fill="white" />
        <line x1="0" y1="68" x2="460" y2="68" stroke="#c8d0dc" stroke-width="1.5" />
        <line x1="0" y1="128" x2="460" y2="128" stroke="#c8d0dc" stroke-width="1.5" />
        <text x="24" y="60" font-size="34" font-family="sans-serif" fill="#18181b">buy milk</text>
        <text x="24" y="120" font-size="34" font-family="sans-serif" fill="#18181b">call dentist</text>
      </svg>
    `;
    const pngBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
    const tempExportRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'snapremind-ocr-stage-test-'));

    try {
      const res = await processImageWithLocalOcr({
        imageBuffer: pngBuffer,
        ocrProfile: 'handwriting',
        exportStages: true,
        exportRootDir: tempExportRoot,
      });

      expect(res.appliedSteps.some((step) => step.startsWith('auto-bbox-shrink'))).toBe(true);
      expect(res.exportDir).toBeDefined();
      const dir = res.exportDir!;
      expect(fs.existsSync(path.join(dir, '03b_background_flattened.png'))).toBe(true);
      expect(fs.existsSync(path.join(dir, '03c_ruled_lines_removed.png'))).toBe(true);
      expect(fs.existsSync(path.join(dir, '04_contrast_enhanced.png'))).toBe(true);
      expect(fs.existsSync(path.join(dir, '05_adaptive_binary.png'))).toBe(true);

      const flattenedMetadata = await sharp(path.join(dir, '03b_background_flattened.png')).metadata();
      expect(flattenedMetadata.width).toBeGreaterThan(0);
      expect(flattenedMetadata.height).toBeGreaterThan(0);
    } finally {
      fs.rmSync(tempExportRoot, { recursive: true, force: true });
    }
  });
});
