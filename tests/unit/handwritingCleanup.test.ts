import { describe, it, expect } from 'vitest';
import {
  flattenBackground,
  removeRuledLines,
  binarizeSauvola,
} from '../../src/lib/ocr/handwriting/cleanup';
import { segmentLines } from '../../src/lib/ocr/handwriting/lines';
import { detectHandwriting } from '../../src/lib/ocr/handwriting/detect';
import { postprocessHandwrittenLines } from '../../src/lib/ocr/postprocess';

function variance(values: number[]): number {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
}

describe('handwriting image cleanup & segmentation', () => {
  it('flattenBackground reduces uneven lighting variance by >80% while preserving ink contrast', () => {
    const w = 240;
    const h = 160;
    const gray = new Uint8ClampedArray(w * h);

    // Gradient shadow from bright (245) on top-left to dark shadow (95) on bottom-right
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const bg = Math.round(245 - (x / w) * 80 - (y / h) * 70);
        gray[y * w + x] = bg;
      }
    }

    // Draw dark ink stroke in the shadowed quadrant
    for (let y = 110; y < 120; y++) {
      for (let x = 140; x < 200; x++) {
        gray[y * w + x] = 25;
      }
    }

    const bgBefore: number[] = [];
    for (let y = 10; y < 80; y += 5) {
      for (let x = 10; x < 220; x += 5) {
        bgBefore.push(gray[y * w + x]);
      }
    }

    const flattened = flattenBackground(gray, w, h);

    const bgAfter: number[] = [];
    for (let y = 10; y < 80; y += 5) {
      for (let x = 10; x < 220; x += 5) {
        bgAfter.push(flattened[y * w + x]);
      }
    }

    const varBefore = variance(bgBefore);
    const varAfter = variance(bgAfter);
    expect(varAfter).toBeLessThan(varBefore * 0.2);

    // Ink inside shadowed region stays dark, paper nearby stays bright
    expect(flattened[115 * w + 170]).toBeLessThan(90);
    expect(flattened[95 * w + 170]).toBeGreaterThan(220);
  });

  it('removeRuledLines erases thin horizontal notebook lines while preserving darker crossing strokes', () => {
    const w = 200;
    const h = 100;
    const gray = new Uint8ClampedArray(w * h).fill(248);

    // Add two horizontal ruled lines at y=35..36 and y=70..71 (gray ~175)
    for (const ry of [35, 36, 70, 71]) {
      for (let x = 0; x < w; x++) {
        gray[ry * w + x] = 175;
      }
    }

    // Add vertical dark ink stroke (value 25) crossing both ruled lines at x=90..94
    for (let y = 20; y < 85; y++) {
      for (let x = 90; x <= 94; x++) {
        gray[y * w + x] = 25;
      }
    }

    const cleaned = removeRuledLines(gray, w, h);

    // Ruled line away from ink should be lifted back to paper brightness
    expect(cleaned[35 * w + 25]).toBeGreaterThan(235);
    expect(cleaned[70 * w + 160]).toBeGreaterThan(235);

    // Crossing dark pen stroke must remain dark
    expect(cleaned[35 * w + 92]).toBeLessThan(50);
    expect(cleaned[70 * w + 92]).toBeLessThan(50);
  });

  it('segmentLines finds exact N lines even with overlapping ascenders/descenders', () => {
    const w = 260;
    const h = 180;
    const binary = new Uint8ClampedArray(w * h).fill(255);

    // 3 text lines centered at y=35, y=85, y=135
    const centers = [35, 85, 135];
    for (const cy of centers) {
      for (let y = cy - 6; y <= cy + 6; y++) {
        for (let x = 25; x <= 230; x++) {
          if (x % 14 < 10) {
            binary[y * w + x] = 0;
          }
        }
      }
    }

    // Add a tall descender from line 1 touching ascender region of line 2 (around x=100)
    for (let y = 41; y <= 79; y++) {
      binary[y * w + 100] = 0;
      binary[y * w + 101] = 0;
    }

    const boxes = segmentLines(binary, w, h);
    expect(boxes.length).toBe(3);
    expect(boxes[0].y1).toBeLessThan(boxes[1].y1);
    expect(boxes[1].y1).toBeLessThan(boxes[2].y1);
  });

  it('detectHandwriting separates uniform printed blocks from joined cursive strokes', () => {
    const w = 300;
    const h = 100;

    // Synthetic printed image: uniform narrow vertical glyphs with consistent height & baseline
    const printed = new Uint8ClampedArray(w * h).fill(255);
    for (let x = 20; x < 260; x += 14) {
      for (let y = 35; y <= 55; y++) {
        for (let dx = 0; dx < 8; dx++) {
          printed[y * w + (x + dx)] = 0;
        }
      }
    }
    const printedRes = detectHandwriting(printed, w, h);
    expect(printedRes.isHandwriting).toBe(false);

    // Synthetic cursive handwriting: wide joined multi-letter components with varying heights & wobble
    const cursive = new Uint8ClampedArray(w * h).fill(255);
    const words = [
      { x0: 15, x1: 95, y0: 32, y1: 54 },
      { x0: 115, x1: 185, y0: 38, y1: 68 },
      { x0: 205, x1: 280, y0: 28, y1: 52 },
    ];
    for (const wd of words) {
      for (let y = wd.y0; y <= wd.y1; y++) {
        for (let x = wd.x0; x <= wd.x1; x++) {
          if (Math.abs(y - (wd.y0 + wd.y1) / 2) <= 6) {
            cursive[y * w + x] = 0;
          }
        }
      }
    }
    const cursiveRes = detectHandwriting(cursive, w, h);
    expect(cursiveRes.isHandwriting).toBe(true);
    expect(cursiveRes.score).toBeGreaterThanOrEqual(0.42);
  });

  it('binarizeSauvola & postprocessHandwrittenLines clean spacing and snap only low-confidence tokens', () => {
    const w = 80;
    const h = 40;
    const gray = new Uint8ClampedArray(w * h).fill(240);
    for (let y = 15; y < 25; y++) {
      for (let x = 20; x < 60; x++) {
        gray[y * w + x] = 30;
      }
    }
    const bin = binarizeSauvola(gray, w, h, 8);
    expect(bin[20 * w + 40]).toBe(0);
    expect(bin[5 * w + 5]).toBe(255);

    const lines = postprocessHandwrittenLines([
      { text: 'call  Dr. Lee   at 3 pm  on 5 / 12 !', confidence: 0.85 },
      { text: 'buy milkk tomorow', confidence: 0.45 },
      { text: 'Bramwell invoice', confidence: 0.45 },
    ]);

    expect(lines[0].text).toBe('call Dr. Lee at 3pm on 5/12!');
    // High-frequency lexicon words at low confidence snap cleanly
    expect(lines[1].text).toBe('buy milk tomorrow');
    // Capitalized proper noun stays intact
    expect(lines[2].text).toContain('Bramwell');
  });
});
