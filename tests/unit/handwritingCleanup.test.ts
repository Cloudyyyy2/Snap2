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
      { text: 'of the Government of the Government of the Government', confidence: 0.78 },
      { text: '0 000 000 000 000 000', confidence: 0.78 },
      { text: 'a b c d e f', confidence: 0.78 },
    ]);

    expect(lines.length).toBe(3);
    expect(lines[0].text).toBe('call Dr. Lee at 3pm on 5/12!');
    // High-frequency lexicon words at low confidence snap cleanly
    expect(lines[1].text).toBe('buy milk tomorrow');
    // Capitalized proper noun stays intact
    expect(lines[2].text).toContain('Bramwell');
  });

  it('autoInvertIfDarkMode flips dark-mode screenshots while leaving light paper unchanged', async () => {
    const { autoInvertIfDarkMode, filterTextComponents } = await import(
      '../../src/lib/ocr/handwriting/cleanup'
    );
    const { splitWideLineBox } = await import('../../src/lib/ocr/handwriting/lines');

    const w = 160;
    const h = 80;
    const darkShot = new Uint8ClampedArray(w * h).fill(22);
    for (let y = 30; y <= 45; y++) {
      for (let x = 30; x <= 120; x++) {
        darkShot[y * w + x] = 235;
      }
    }
    const resDark = autoInvertIfDarkMode(darkShot, w, h);
    expect(resDark.inverted).toBe(true);
    expect(resDark.data[5 * w + 5]).toBeGreaterThan(220);
    expect(resDark.data[38 * w + 60]).toBeLessThan(30);

    const lightPaper = new Uint8ClampedArray(w * h).fill(245);
    lightPaper[38 * w + 60] = 20;
    const resLight = autoInvertIfDarkMode(lightPaper, w, h);
    expect(resLight.inverted).toBe(false);

    // filterTextComponents removes giant border clutter while keeping centered ink
    const bin = new Uint8ClampedArray(300 * 220).fill(255);
    for (let y = 0; y < 180; y++) {
      for (let x = 0; x < 40; x++) {
        bin[y * 300 + x] = 0;
      }
    }
    for (let y = 100; y <= 116; y++) {
      for (let x = 120; x <= 145; x++) {
        bin[y * 300 + x] = 0;
      }
    }
    const filtered = filterTextComponents(bin, 300, 220);
    expect(filtered[40 * 300 + 15]).toBe(255);
    expect(filtered[108 * 300 + 132]).toBe(0);

    // splitWideLineBox splits overly wide lines (> 5:1) at whitespace gaps
    const wideW = 500;
    const wideH = 60;
    const wideBin = new Uint8ClampedArray(wideW * wideH).fill(255);
    for (let y = 20; y <= 45; y++) {
      for (let x = 20; x <= 180; x++) wideBin[y * wideW + x] = 0;
      for (let x = 230; x <= 460; x++) wideBin[y * wideW + x] = 0;
    }
    const chunks = splitWideLineBox(wideBin, wideW, wideH, {
      x0: 20,
      y0: 20,
      x1: 460,
      y1: 45,
    });
    expect(chunks.length).toBeGreaterThan(1);
  });

  it('runs Stage 1-4 enhancements: Otsu, median 3x3, bilateral smoothing, morphology, border crop, quad warp, DPI scale, and XY-cut layout regions', async () => {
    const {
      binarizeOtsu,
      applyMedianFilter3x3,
      applyBilateralSmoothing,
      applyMorphologicalOp,
      cropDarkBordersAndMargins,
      warpPerspectiveQuad,
      computeRecommendedDpiScale,
    } = await import('../../src/lib/ocr/handwriting/cleanup');
    const { segmentLayoutRegions } = await import('../../src/lib/ocr/handwriting/layout');

    // 1. Otsu binarization separates bimodal grayscale cleanly
    const bimodal = new Uint8ClampedArray(60 * 40).fill(235);
    for (let y = 10; y < 30; y++) {
      for (let x = 15; x < 45; x++) bimodal[y * 60 + x] = 35;
    }
    const otsu = binarizeOtsu(bimodal, 60, 40);
    expect(otsu[20 * 60 + 30]).toBe(0);
    expect(otsu[2 * 60 + 2]).toBe(255);

    // 2. Median 3x3 removes isolated salt-and-pepper specks while keeping solid edges sharp
    const noisy = new Uint8ClampedArray(50 * 50).fill(245);
    for (let y = 20; y < 30; y++) {
      for (let x = 20; x < 30; x++) noisy[y * 50 + x] = 20;
    }
    noisy[5 * 50 + 5] = 0; // isolated black speck
    noisy[25 * 50 + 25] = 255; // isolated white pinhole inside ink
    const denoised = applyMedianFilter3x3(noisy, 50, 50);
    expect(denoised[5 * 50 + 5]).toBe(245);
    expect(denoised[25 * 50 + 25]).toBe(20);

    // 3. Bilateral smoothing reduces subtle background ripple without washing out strong ink step
    const rippled = new Uint8ClampedArray(40 * 40);
    for (let y = 0; y < 40; y++) {
      for (let x = 0; x < 40; x++) {
        rippled[y * 40 + x] = x >= 20 ? 30 : 230 + ((x + y) % 2 === 0 ? 10 : -10);
      }
    }
    const smoothed = applyBilateralSmoothing(rippled, 40, 40, 2.0, 25.0);
    expect(smoothed[20 * 40 + 10]).toBeGreaterThan(215);
    expect(smoothed[20 * 40 + 30]).toBeLessThan(45);

    // 4. Morphological closing bridges a 1px horizontal gap in broken ink; opening removes 1px spur
    const broken = new Uint8ClampedArray(30 * 20).fill(255);
    for (let x = 5; x <= 10; x++) broken[10 * 30 + x] = 0;
    for (let x = 12; x <= 18; x++) broken[10 * 30 + x] = 0;
    broken[2 * 30 + 2] = 0; // isolated 1px dot
    const closed = applyMorphologicalOp(broken, 30, 20, 'close', 1);
    expect(closed[10 * 30 + 11]).toBe(0);
    const opened = applyMorphologicalOp(broken, 30, 20, 'open', 1);
    expect(opened[2 * 30 + 2]).toBe(255);

    // 5. cropDarkBordersAndMargins trims dark desk margins around bright central paper
    const cw = 200;
    const ch = 120;
    const bordered = new Uint8ClampedArray(cw * ch).fill(45);
    for (let y = 15; y < 105; y++) {
      for (let x = 40; x < 160; x++) bordered[y * cw + x] = 242;
    }
    const cropRes = cropDarkBordersAndMargins(bordered, cw, ch);
    expect(cropRes.cropped).toBe(true);
    expect(cropRes.width).toBeLessThan(cw * 0.75);
    expect(cropRes.height).toBeLessThan(ch * 0.85);

    // 6. warpPerspectiveQuad flattens a tilted trapezoid onto an upright rectangle
    const qw = 160;
    const qh = 120;
    const quadSrc = new Uint8ClampedArray(qw * qh).fill(30);
    for (let y = 10; y <= 110; y++) {
      const t = (y - 10) / 100;
      const lx = Math.round(30 - t * 20);
      const rx = Math.round(130 + t * 20);
      for (let x = lx; x <= rx; x++) quadSrc[y * qw + x] = 240;
    }
    const warped = warpPerspectiveQuad(quadSrc, qw, qh, [
      { x: 30, y: 10 },
      { x: 130, y: 10 },
      { x: 150, y: 110 },
      { x: 10, y: 110 },
    ]);
    expect(warped.width).toBeGreaterThanOrEqual(135);
    expect(warped.data[Math.floor(warped.height / 2) * warped.width + Math.floor(warped.width / 2)]).toBeGreaterThan(220);

    // 7. computeRecommendedDpiScale upsamples tiny text runs (>= 1.5x)
    const tiny = new Uint8ClampedArray(300 * 100).fill(255);
    for (let x = 20; x < 280; x += 4) {
      for (let y = 40; y < 48; y++) tiny[y * 300 + x] = 0; // 8px tiny glyphs
    }
    expect(computeRecommendedDpiScale(tiny, 300, 100)).toBeGreaterThanOrEqual(2.0);

    // 8. segmentLayoutRegions separates 2 distinct columns in left-to-right reading order
    const rw = 320;
    const rh = 160;
    const twoCol = new Uint8ClampedArray(rw * rh).fill(255);
    for (let y = 25; y <= 125; y++) {
      for (let x = 20; x <= 110; x++) twoCol[y * rw + x] = 0;
      for (let x = 200; x <= 295; x++) twoCol[y * rw + x] = 0;
    }
    const regions = segmentLayoutRegions(twoCol, rw, rh);
    expect(regions.length).toBe(2);
    expect(regions[0].x1).toBeLessThan(regions[1].x0);
  });

  it('cropDarkBordersAndMargins preserves central text even when an interior vertical rule or dense column lowers middle column brightness', async () => {
    const { cropDarkBordersAndMargins } = await import(
      '../../src/lib/ocr/handwriting/cleanup'
    );
    const w = 320;
    const h = 220;
    // Outer desk border (35) around shaded indoor paper (175)
    const gray = new Uint8ClampedArray(w * h).fill(35);
    for (let y = 20; y < 200; y++) {
      for (let x = 24; x < 296; x++) {
        gray[y * w + x] = 175;
      }
    }
    // Vertical notebook margin line / heavy stroke column at x=120..124 splitting the sheet
    for (let y = 22; y < 198; y++) {
      for (let x = 120; x <= 124; x++) {
        gray[y * w + x] = 45;
      }
    }
    // Dense writing block on the right side (x=145..260, y=50..160)
    for (let y = 50; y < 160; y++) {
      for (let x = 145; x < 260; x++) {
        if ((y % 18) < 10) gray[y * w + x] = 28;
      }
    }

    const res = cropDarkBordersAndMargins(gray, w, h);
    expect(res.cropped).toBe(true);
    // Must keep both sides of the interior rule (width close to sheet width 272px, never cropped to just blank left margin)
    expect(res.width).toBeGreaterThanOrEqual(250);
    expect(res.bounds.x0).toBeLessThanOrEqual(30);
    expect(res.bounds.x1).toBeGreaterThanOrEqual(285);
  });

  it('segmentWords detects individual word edges and fits each word into a square box', async () => {
    const { segmentWords, cropToSquare } = await import(
      '../../src/lib/ocr/handwriting/cleanup'
    );
    const w = 360;
    const h = 140;
    const binary = new Uint8ClampedArray(w * h).fill(255);

    // Line 1: Word 1 "BUY" at x=20..75, y=30..55; Word 2 "MILK" at x=110..185, y=30..55
    for (let y = 30; y <= 55; y++) {
      for (let x = 20; x <= 75; x++) binary[y * w + x] = 0;
      for (let x = 110; x <= 185; x++) binary[y * w + x] = 0;
    }

    // Line 2: Word 3 "CALL" at x=25..95, y=85..110; Word 4 "NOW" at x=140..210, y=85..110
    for (let y = 85; y <= 110; y++) {
      for (let x = 25; x <= 95; x++) binary[y * w + x] = 0;
      for (let x = 140; x <= 210; x++) binary[y * w + x] = 0;
    }

    const words = segmentWords(binary, w, h);
    expect(words.length).toBe(4);

    // Word 1 checks
    expect(words[0].x0).toBe(20);
    expect(words[0].x1).toBe(75);
    expect(words[0].y0).toBe(30);
    expect(words[0].y1).toBe(55);

    // Word 1 squareBox must be an exact square
    const sq1 = words[0].squareBox;
    expect(sq1.size).toBeGreaterThanOrEqual(words[0].x1 - words[0].x0 + 1);
    expect(sq1.x1 - sq1.x0 + 1).toBe(sq1.size);
    expect(sq1.y1 - sq1.y0 + 1).toBe(sq1.size);

    // cropToSquare must produce a square Uint8ClampedArray with centered ink and white padding
    const gray = new Uint8ClampedArray(binary);
    const croppedSq = cropToSquare(gray, w, h, words[0], 128);
    expect(croppedSq.size).toBe(128);
    expect(croppedSq.data.length).toBe(128 * 128);

    // Center pixel should be ink (0), corner pixel should be paper padding (255)
    const midIdx = 64 * 128 + 64;
    expect(croppedSq.data[midIdx]).toBe(0);
    expect(croppedSq.data[0]).toBe(255);
  });

  it('filterTextComponents strips periodic margin spiral coils and dark footer bars while keeping handwriting', async () => {
    const { filterTextComponents } = await import('../../src/lib/ocr/handwriting/cleanup');
    const w = 320;
    const h = 240;
    const bin = new Uint8ClampedArray(w * h).fill(255);

    // 6 periodic spiral-binding coil loops along left margin (x=8..20, pitch 34px)
    for (const cy of [20, 54, 88, 122, 156, 190]) {
      for (let y = cy; y < cy + 12; y++) {
        for (let x = 8; x <= 20; x++) {
          bin[y * w + x] = 0;
        }
      }
    }

    // Dark horizontal bottom watermark bar touching bottom edge (width > 0.5*w)
    for (let y = 233; y < h; y++) {
      for (let x = 35; x <= 285; x++) {
        bin[y * w + x] = 0;
      }
    }

    // Interior handwriting words away from spiral coil continuation
    for (let y = 70; y <= 84; y++) {
      for (let x = 80; x <= 170; x++) bin[y * w + x] = 0;
    }
    for (let y = 135; y <= 150; y++) {
      for (let x = 85; x <= 185; x++) bin[y * w + x] = 0;
    }

    const filtered = filterTextComponents(bin, w, h);

    // Spiral coils & bottom watermark bar are stripped
    expect(filtered[26 * w + 14]).toBe(255);
    expect(filtered[94 * w + 14]).toBe(255);
    expect(filtered[236 * w + 160]).toBe(255);

    // Interior handwriting ink is preserved
    expect(filtered[77 * w + 120]).toBe(0);
    expect(filtered[142 * w + 130]).toBe(0);
  });

  it('segmentLines caps vertical padding at midpoint valleys so tightly spaced lines never overlap', () => {
    const w = 320;
    const h = 200;
    const bin = new Uint8ClampedArray(w * h).fill(255);

    // 3 tightly spaced lines at y=45, y=80, y=115 (13px tall each)
    for (const cy of [45, 80, 115]) {
      for (let y = cy - 6; y <= cy + 6; y++) {
        for (let x = 60; x <= 260; x++) {
          if (x % 16 < 11) bin[y * w + x] = 0;
        }
      }
    }

    const boxes = segmentLines(bin, w, h);
    expect(boxes.length).toBe(3);
    expect(boxes[0].y1).toBeLessThan(boxes[1].y0);
    expect(boxes[1].y1).toBeLessThan(boxes[2].y0);
  });

  it('deslantGrayscale estimates and straightens forward-slanted handwriting strokes', async () => {
    const { deslantGrayscale } = await import('../../src/lib/ocr/handwriting/cleanup');
    const w = 96;
    const h = 48;
    const gray = new Uint8ClampedArray(w * h).fill(255);

    // Draw slanted strokes leaning rightward toward top (+20 deg shear)
    for (const baseX of [24, 48, 72]) {
      for (let y = 8; y <= 38; y++) {
        const cx = Math.round(baseX - (y - 23) * 0.36);
        for (let dx = -1; dx <= 1; dx++) {
          gray[y * w + cx + dx] = 15;
        }
      }
    }

    const res = deslantGrayscale(gray, w, h);
    expect(res.corrected).toBe(true);
    expect(Math.abs(res.slantDeg)).toBeGreaterThanOrEqual(8);
  });

  it('stripLineEdgeBleeds keeps thin right-edge letter stems near words while dropping isolated top-edge bleeds', async () => {
    const { stripLineEdgeBleeds, segmentHandwrittenWordsInLine } = await import(
      '../../src/lib/ocr/handwriting/lines'
    );
    const w = 140;
    const h = 44;
    const bin = new Uint8ClampedArray(w * h).fill(255);
    const gray = new Uint8ClampedArray(w * h).fill(255);

    // Isolated top-edge descender bleed from previous line at x=2..4, y=0..24
    for (let y = 0; y <= 24; y++) {
      for (let x = 2; x <= 4; x++) {
        bin[y * w + x] = 0;
        gray[y * w + x] = 30;
      }
    }

    // Core word glyphs at x=28..120, y=10..34 with faint gray bridge across x=68..75
    for (let y = 10; y <= 34; y++) {
      for (let x = 28; x <= 67; x++) {
        bin[y * w + x] = 0;
        gray[y * w + x] = 25;
      }
      for (let x = 68; x <= 75; x++) {
        gray[y * w + x] = 195; // faint pen stroke missed by binary threshold
      }
      for (let x = 76; x <= 120; x++) {
        bin[y * w + x] = 0;
        gray[y * w + x] = 25;
      }
      // Thin right-edge letter stem ('i') only 5px from main word
      if (y >= 12 && y <= 32) {
        for (let x = 125; x <= 128; x++) {
          bin[y * w + x] = 0;
          gray[y * w + x] = 25;
        }
      }
    }

    const box = { x0: 0, y0: 0, x1: w - 1, y1: h - 1 };
    const { mask } = stripLineEdgeBleeds(bin, w, h, box);
    // Isolated top-edge bleed dropped
    expect(mask[10 * w + 3]).toBe(255);
    // Right-edge letter stem preserved
    expect(mask[20 * w + 126]).toBe(0);

    // Faint gray stroke bridges x=67..76 so the entire word stays one crop
    const crops = segmentHandwrittenWordsInLine(bin, gray, w, h, box, 2);
    expect(crops.length).toBe(1);
    expect(crops[0].box.x0).toBe(28);
    expect(crops[0].box.x1).toBe(128);
  });
});

