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
    expect(isHandwritingModelAvailable()).toBe(true);
    expect(getActiveHandwritingModelName()).toBe('trocr-base-handwritten');
    expect(isHandwritingModelAvailable('/tmp/nonexistent-trocr-model-dir')).toBe(false);
    expect(isPrintedModelAvailable()).toBe(true);
    expect(getActivePrintedModelName()).toBe('tessdata_full');
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
});
