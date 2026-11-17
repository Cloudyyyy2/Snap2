import fs from 'fs';
import os from 'os';
import path from 'path';
import sharp from 'sharp';
import { describe, it, expect } from 'vitest';
import {
  isHandwritingModelAvailable,
  processImageWithLocalOcr,
} from '../../server/handwritingRecognizer';
import { parseReminderLocally } from '../../server/localReminderParser';

describe('local handwriting TrOCR recognizer & offline parser', () => {
  it('detects downloaded local model and rejects nonexistent model directory', () => {
    expect(isHandwritingModelAvailable()).toBe(true);
    expect(isHandwritingModelAvailable('/tmp/nonexistent-trocr-model-dir')).toBe(false);
  });

  it(
    'runs TrOCR offline on synthetic handwritten-style PNG and writes stage exports',
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

        const dir = res.exportDir!;
        expect(fs.existsSync(path.join(dir, '03b_background_flattened.png'))).toBe(true);
        expect(fs.existsSync(path.join(dir, '03c_ruled_lines_removed.png'))).toBe(true);
        expect(fs.existsSync(path.join(dir, '06_line_01.png'))).toBe(true);
        expect(fs.existsSync(path.join(dir, '06_line_02.png'))).toBe(true);

        const report = fs.readFileSync(path.join(dir, 'extracted_text.txt'), 'utf8');
        expect(report).toContain('Engine: TrOCR');

        // Also confirm printed mode routes to vision-fallback without invoking TrOCR
        const printedRes = await processImageWithLocalOcr({
          imageBuffer: pngBuffer,
          ocrProfile: 'printed',
          exportStages: false,
        });
        expect(printedRes.recognizer).toBe('vision-fallback');
      } finally {
        fs.rmSync(tempExportRoot, { recursive: true, force: true });
      }
    },
    30000
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
