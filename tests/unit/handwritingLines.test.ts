import { describe, it, expect } from 'vitest';
import {
  stripSpiralAndCornerArtifacts,
  filterTextComponents,
} from '../../src/lib/ocr/handwriting/cleanup';
import {
  segmentLines,
  cropLine,
  splitWideLineBox,
} from '../../src/lib/ocr/handwriting/lines';

describe('F5: Spiral/corner artifact cleanup & ascender/descender/ligature-safe line segmentation', () => {
  it('stripSpiralAndCornerArtifacts and segmentLines remove left spiral holes, corner wedges, and top-right border bleeds while preserving wide lines', () => {
    for (const [w, h] of [
      [273, 402],
      [360, 420],
    ]) {
      const bin = new Uint8ClampedArray(w * h).fill(255);

      // 1. 12 left spiral margin holes (x=0..7)
      for (let i = 0; i < 12; i++) {
        const cy = 16 + i * 31;
        for (let y = cy; y < cy + 11 && y < h; y++) {
          for (let x = 0; x <= 7; x++) {
            bin[y * w + x] = 0;
          }
        }
      }

      // 2. Top-right rounded corner wedge (touching x=w-1 and y=0)
      for (let y = 0; y <= 16; y++) {
        for (let x = w - 6; x < w; x++) {
          bin[y * w + x] = 0;
        }
      }

      // 3. Top-right border-clipped partial column bleed ("Who / in" fragment)
      for (const frag of [
        { x0: w - 26, x1: w - 16, y0: 26, y1: 38 },
        { x0: w - 14, x1: w - 2, y0: 24, y1: 40 },
      ]) {
        for (let y = frag.y0; y <= frag.y1; y++) {
          for (let x = frag.x0; x <= frag.x1; x++) {
            bin[y * w + x] = 0;
          }
        }
      }

      // 4. Legitimate main lines: Line 1 "bread" (x=46..102), Line 2 "milk" (x=48..92),
      //    Line 3 wide phrase spanning 0.15*w..0.88*w
      const drawBar = (x0: number, x1: number, y0: number, y1: number) => {
        for (let y = y0; y <= y1; y++) {
          for (let x = x0; x <= x1; x++) {
            if (x % 8 < 5) bin[y * w + x] = 0;
          }
        }
      };
      drawBar(46, 102, 26, 40);
      drawBar(48, 92, 62, 76);
      const wideX0 = Math.round(w * 0.15);
      const wideX1 = Math.round(w * 0.88);
      drawBar(wideX0, wideX1, 110, 126);

      const stripped = stripSpiralAndCornerArtifacts(bin, w, h);
      // Spiral hole & corner bleed pixels stripped
      expect(stripped[20 * w + 3]).toBe(255);
      expect(stripped[5 * w + (w - 3)]).toBe(255);
      expect(stripped[32 * w + (w - 8)]).toBe(255);
      // Legitimate words kept
      expect(stripped[33 * w + 64]).toBe(0);
      expect(stripped[118 * w + Math.round(w * 0.5)]).toBe(0);

      const boxes = segmentLines(bin, w, h);
      expect(boxes.length).toBe(3);
      // Line 1 stays tight around "bread" and is NOT stretched to the top-right bleed
      expect(boxes[0].x0).toBeGreaterThanOrEqual(40);
      expect(boxes[0].x1).toBeLessThanOrEqual(110);
      // Wide line spanning 0.15*w..0.88*w is NOT clipped
      expect(boxes[2].x0).toBeLessThanOrEqual(wideX0 + 4);
      expect(boxes[2].x1).toBeGreaterThanOrEqual(wideX1 - 4);
    }
  });

  it('segmentLines preserves thin 1-2px f/g/j/l/p/q/y ascenders and descenders on tightly spaced multi-line cursive', () => {
    const w = 360;
    const h = 220;
    const bin = new Uint8ClampedArray(w * h).fill(255);

    // 4 tightly spaced cursive lines (centers at y=38, 78, 118, 158; 40px pitch)
    const centers = [38, 78, 118, 158];
    for (const cy of centers) {
      for (let y = cy - 6; y <= cy + 6; y++) {
        for (let x = 40; x <= 260; x++) {
          if (x % 9 < 6) bin[y * w + x] = 0;
        }
      }
      // Thin 2px ascender (like 'l'/'f') extending 15px above body
      for (let y = cy - 21; y < cy - 6; y++) {
        bin[y * w + 80] = 0;
        bin[y * w + 81] = 0;
      }
      // Thin 2px descender (like 'g'/'p'/'y') extending 15px below body
      for (let y = cy + 7; y <= cy + 21; y++) {
        bin[y * w + 150] = 0;
        bin[y * w + 151] = 0;
      }
    }

    const boxes = segmentLines(bin, w, h);
    expect(boxes.length).toBe(4);

    // Strictly non-overlapping vertical bounds
    for (let i = 0; i < boxes.length - 1; i++) {
      expect(boxes[i].y1).toBeLessThan(boxes[i + 1].y0);
    }

    // Outer ascender on line 0 and outer descender on line 3 are fully enclosed
    expect(boxes[0].y0).toBeLessThanOrEqual(38 - 21);
    expect(boxes[3].y1).toBeGreaterThanOrEqual(158 + 21);

    // Interior lines include ascender/descender tips up to valley without clipping
    expect(boxes[1].y0).toBeLessThanOrEqual(78 - 19);
    expect(boxes[1].y1).toBeGreaterThanOrEqual(78 + 19);
  });

  it('cropLine recovers nearby ascender/descender grayscale context, white-pads out-of-bounds margins, and stops before neighbor lines', () => {
    const w = 180;
    const h = 120;
    const gray = new Uint8ClampedArray(w * h).fill(245);

    // Neighbor line 1 at y=10..22
    for (let y = 10; y <= 22; y++) {
      for (let x = 30; x <= 140; x++) gray[y * w + x] = 25;
    }

    // Target line 2 core at y=40..54, plus an ascender tip at y=36..39 (just above box.y0=40)
    // Separated from neighbor line 1 by clean white valley y=23..35 (13px gap)
    for (let y = 40; y <= 54; y++) {
      for (let x = 30; x <= 140; x++) gray[y * w + x] = 25;
    }
    for (let y = 36; y <= 39; y++) {
      gray[y * w + 70] = 30;
      gray[y * w + 71] = 30;
    }

    const pad = 16;
    const crop = cropLine(gray, w, h, { x0: 30, y0: 40, x1: 140, y1: 54 }, pad);
    expect(crop.width).toBe(140 - 30 + 1 + pad * 2);
    expect(crop.height).toBe(54 - 40 + 1 + pad * 2);

    // Ascender tip at global y=37 (crop local y = 37 - 40 + 16 = 13, x = 70 - 30 + 16 = 56) is preserved!
    expect(crop.data[13 * crop.width + 56]).toBeLessThan(60);

    // Neighbor line 1 at global y=20 (crop local y = 20 - 40 + 16 = -4 -> not in crop, or global y=22 -> local y=-2)
    // Test with larger pad=24 where neighbor line WOULD fall inside pad: must still be stopped!
    const largePad = 24;
    const cropLarge = cropLine(gray, w, h, { x0: 30, y0: 40, x1: 140, y1: 54 }, largePad);
    // Neighbor line row global y=20 -> local y = 20 - 40 + 24 = 4; must remain white (255)!
    expect(cropLarge.data[4 * cropLarge.width + 50]).toBe(255);
    // Ascender tip at global y=37 -> local y = 37 - 40 + 24 = 21; still preserved!
    expect(cropLarge.data[21 * cropLarge.width + (70 - 30 + largePad)]).toBeLessThan(60);

    // Out-of-bounds top margin (box.y0 = 2, pad = 16) stays white (255)
    const edgeCrop = cropLine(gray, w, h, { x0: 30, y0: 2, x1: 140, y1: 22 }, 16);
    expect(edgeCrop.data[0]).toBe(255);
    expect(edgeCrop.data[10 * edgeCrop.width + 50]).toBe(255);
  });

  it('splitWideLineBox splits at whitespace gaps, splits connected cursive strictly inside thin ligature valleys, and aborts safely on solid dense words', () => {
    // 1. Whitespace gap in extended lookahead (maxAspect = 5.0, boxH = 30 => maxChunkW = 150)
    const w1 = 420;
    const h1 = 60;
    const binGap = new Uint8ClampedArray(w1 * h1).fill(255);
    for (let y = 15; y <= 44; y++) {
      for (let x = 20; x <= 185; x++) binGap[y * w1 + x] = 0; // width 166 > maxChunkW (150), <= 1.25*150 (187)
      for (let x = 200; x <= 380; x++) binGap[y * w1 + x] = 0;
    }
    const gapChunks = splitWideLineBox(binGap, w1, h1, { x0: 20, y0: 15, x1: 380, y1: 44 }, 5.0);
    expect(gapChunks.length).toBe(2);
    expect(gapChunks[0].x1).toBeGreaterThanOrEqual(185);
    expect(gapChunks[0].x1).toBeLessThan(200);
    expect(gapChunks[1].x0).toBeGreaterThanOrEqual(200);

    // 2. Connected cursive words bridged by a thin 2px-tall ligature between tall stems (x=135..148)
    const w2 = 380;
    const h2 = 60;
    const binLig = new Uint8ClampedArray(w2 * h2).fill(255);
    for (let y = 12; y <= 45; y++) {
      for (let x = 10; x <= 134; x++) binLig[y * w2 + x] = 0;
      for (let x = 150; x <= 340; x++) binLig[y * w2 + x] = 0;
    }
    // Thin 2px horizontal cursive ligature connecting across x=135..149 at y=36..37
    for (let y = 36; y <= 37; y++) {
      for (let x = 135; x <= 149; x++) binLig[y * w2 + x] = 0;
    }
    const ligChunks = splitWideLineBox(binLig, w2, h2, { x0: 10, y0: 12, x1: 340, y1: 45 }, 5.0);
    expect(ligChunks.length).toBe(2);
    // Split must land strictly inside the ligature bridge (x=135..149), never slicing tall stems
    expect(ligChunks[0].x1).toBeGreaterThanOrEqual(135);
    expect(ligChunks[0].x1).toBeLessThanOrEqual(149);

    // 3. Dense unbroken solid block with zero gaps and zero ligature valleys: refuses to slice mid-character
    const w3 = 320;
    const h3 = 50;
    const binDense = new Uint8ClampedArray(w3 * h3).fill(255);
    for (let y = 10; y <= 39; y++) {
      for (let x = 10; x <= 290; x++) binDense[y * w3 + x] = 0;
    }
    const denseChunks = splitWideLineBox(binDense, w3, h3, { x0: 10, y0: 10, x1: 290, y1: 39 }, 5.0);
    expect(denseChunks.length).toBe(1);
    expect(denseChunks[0]).toEqual({ x0: 10, y0: 10, x1: 290, y1: 39 });
  });

  it('preserves bulleted lists, numbered lists with 2x2 dots, right-aligned prices, and narrow right columns while stripping corner border bars', () => {
    // 1. Bulleted lists at x=6 and x=18 (5 items each)
    for (const bx of [6, 18]) {
      const w = 320;
      const h = 240;
      const bin = new Uint8ClampedArray(w * h).fill(255);
      for (let i = 0; i < 5; i++) {
        const cy = 30 + i * 42;
        for (let y = cy; y <= cy + 4; y++) {
          for (let x = bx; x <= bx + 4; x++) bin[y * w + x] = 0;
        }
        for (let y = cy - 4; y <= cy + 8; y++) {
          for (let x = 60; x <= 220; x++) {
            if (x % 8 < 5) bin[y * w + x] = 0;
          }
        }
      }
      const stripped = stripSpiralAndCornerArtifacts(bin, w, h);
      const filtered = filterTextComponents(bin, w, h);
      const boxes = segmentLines(bin, w, h);
      expect(boxes.length).toBe(5);
      for (let i = 0; i < 5; i++) {
        const cy = 30 + i * 42;
        expect(stripped[(cy + 2) * w + (bx + 2)]).toBe(0);
        expect(filtered[(cy + 2) * w + (bx + 2)]).toBe(0);
        expect(boxes[i].x0).toBeLessThanOrEqual(bx + 2);
      }
    }

    // 2. Numbered list ("1.".."5.") with 2x2 period dots (area=4)
    {
      const w = 340;
      const h = 260;
      const bin = new Uint8ClampedArray(w * h).fill(255);
      for (let i = 0; i < 5; i++) {
        const cy = 30 + i * 44;
        for (let y = cy; y <= cy + 12; y++) {
          for (let x = 16; x <= 20; x++) bin[y * w + x] = 0;
        }
        for (let y = cy + 11; y <= cy + 12; y++) {
          for (let x = 23; x <= 24; x++) bin[y * w + x] = 0;
        }
        for (let y = cy; y <= cy + 12; y++) {
          for (let x = 55; x <= 220; x++) {
            if (x % 8 < 5) bin[y * w + x] = 0;
          }
        }
      }
      const filtered = filterTextComponents(bin, w, h);
      const boxes = segmentLines(bin, w, h);
      expect(boxes.length).toBe(5);
      for (let i = 0; i < 5; i++) {
        const cy = 30 + i * 44;
        expect(filtered[(cy + 6) * w + 18]).toBe(0);
        expect(filtered[(cy + 11) * w + 23]).toBe(0);
        expect(boxes[i].x0).toBeLessThanOrEqual(18);
      }
    }

    // 3. Right-aligned prices ($12.50 at w-2, $9.50 at w-10) and narrow right column (x=335..395 on w=400)
    for (const [w, px0, px1] of [
      [360, 320, 358],
      [400, 356, 390],
      [400, 335, 395],
    ]) {
      const h = 240;
      const bin = new Uint8ClampedArray(w * h).fill(255);
      for (let i = 0; i < 4; i++) {
        const cy = 30 + i * 48;
        for (let y = cy; y <= cy + 12; y++) {
          for (let x = 25; x <= 140; x++) {
            if (x % 8 < 5) bin[y * w + x] = 0;
          }
          for (let x = px0; x <= px1; x++) {
            if ((x - px0) % 7 < 5) bin[y * w + x] = 0;
          }
        }
      }
      const stripped = stripSpiralAndCornerArtifacts(bin, w, h);
      const filtered = filterTextComponents(bin, w, h);
      const boxes = segmentLines(bin, w, h);
      expect(boxes.length).toBe(4);
      for (let i = 0; i < 4; i++) {
        const cy = 30 + i * 48;
        expect(stripped[(cy + 6) * w + px0]).toBe(0);
        expect(filtered[(cy + 6) * w + px0]).toBe(0);
        expect(boxes[i].x1).toBeGreaterThanOrEqual(px1 - 4);
      }
    }

    // 4. Bottom-right corner border bar (e.g. 58x6 touching bottom & right edge) is stripped cleanly
    {
      const w = 273;
      const h = 398;
      const bin = new Uint8ClampedArray(w * h).fill(255);
      for (let y = 368; y <= 386; y++) {
        for (let x = 67; x <= 135; x++) {
          if (x % 6 < 4) bin[y * w + x] = 0;
        }
      }
      for (let y = h - 6; y < h; y++) {
        bin[y * w + (w - 1)] = 0;
      }
      for (let y = h - 2; y < h; y++) {
        for (let x = w - 58; x < w; x++) bin[y * w + x] = 0;
      }
      const stripped = stripSpiralAndCornerArtifacts(bin, w, h);
      expect(stripped[(h - 1) * w + (w - 20)]).toBe(255);
      const boxes = segmentLines(stripped, w, h);
      expect(boxes.length).toBe(1);
      expect(boxes[0].x1).toBeLessThanOrEqual(140);
    }
  });

  it('cropLine prevents neighbor-line bleed across 3-4px gaps with 2px neighbor tips and recovers crossing valley tips without neighbor bodies', () => {
    // 1. Tight 4px gap + 2px neighbor descender tip (Probe 3c)
    {
      const w = 180;
      const h = 90;
      const gray = new Uint8ClampedArray(w * h).fill(255);
      // Neighbor line 1 core at y=10..22
      for (let y = 10; y <= 22; y++) {
        for (let x = 30; x <= 140; x++) gray[y * w + x] = 25;
      }
      // 2px neighbor descender tip at y=23..26
      for (let y = 23; y <= 26; y++) {
        gray[y * w + 70] = 30;
        gray[y * w + 71] = 30;
      }
      // 4px clean white gap at y=27..30; target line 2 at y=31..45
      for (let y = 31; y <= 45; y++) {
        for (let x = 30; x <= 140; x++) gray[y * w + x] = 25;
      }

      const crop = cropLine(gray, w, h, { x0: 30, y0: 31, x1: 140, y1: 45 }, 16);
      // Neighbor core (global y=15..22 -> crop local y=0..7) must remain white (255)
      for (let ly = 0; ly <= 7; ly++) {
        for (let lx = 0; lx < crop.width; lx++) {
          expect(crop.data[ly * crop.width + lx]).toBe(255);
        }
      }
    }

    // 2. Crossing ascender/descender across valley midpoint (40px pitch, Probe 3a)
    {
      const w = 360;
      const h = 140;
      const bin = new Uint8ClampedArray(w * h).fill(255);
      // Line 0 body y=21..47, descender at x=150..151 down to y=66
      for (let y = 21; y <= 47; y++) {
        for (let x = 40; x <= 260; x++) {
          if (x % 9 < 6) bin[y * w + x] = 0;
        }
      }
      for (let y = 48; y <= 66; y++) {
        bin[y * w + 150] = 0;
        bin[y * w + 151] = 0;
      }
      // Line 1 ascender at x=80..81 up to y=54, body y=74..99
      for (let y = 54; y <= 73; y++) {
        bin[y * w + 80] = 0;
        bin[y * w + 81] = 0;
      }
      for (let y = 74; y <= 99; y++) {
        for (let x = 40; x <= 260; x++) {
          if (x % 9 < 6) bin[y * w + x] = 0;
        }
      }

      const boxes = segmentLines(bin, w, h);
      expect(boxes.length).toBe(2);
      expect(boxes[0].y1).toBeLessThan(boxes[1].y0);

      // cropLine on Line 0 recovers descender tip at y=65 without pulling Line 1's dense body (y=74..99)
      const pad = 16;
      const crop0 = cropLine(bin, w, h, boxes[0], pad);
      const localTipY = 65 - boxes[0].y0 + pad;
      const localTipX = 150 - boxes[0].x0 + pad;
      expect(crop0.data[localTipY * crop0.width + localTipX]).toBe(0);

      const localNeighborBodyY = 76 - boxes[0].y0 + pad;
      if (localNeighborBodyY >= 0 && localNeighborBodyY < crop0.height) {
        let neighborDark = 0;
        for (let lx = 0; lx < crop0.width; lx++) {
          if (crop0.data[localNeighborBodyY * crop0.width + lx] < 180) neighborDark++;
        }
        expect(neighborDark).toBe(0);
      }

      // cropLine on Line 1 recovers ascender tip at y=55 without pulling Line 0's dense body (y=21..47)
      const crop1 = cropLine(bin, w, h, boxes[1], pad);
      const localAscY = 55 - boxes[1].y0 + pad;
      const localAscX = 80 - boxes[1].x0 + pad;
      expect(crop1.data[localAscY * crop1.width + localAscX]).toBe(0);

      const localTopBodyY = 44 - boxes[1].y0 + pad;
      if (localTopBodyY >= 0 && localTopBodyY < crop1.height) {
        let topDark = 0;
        for (let lx = 0; lx < crop1.width; lx++) {
          if (crop1.data[localTopBodyY * crop1.width + lx] < 180) topDark++;
        }
        expect(topDark).toBe(0);
      }
    }
  });

  it('splitWideLineBox refuses to slice hollow loops on padded LineBoxes and splits 2px ligatures between tall stems', () => {
    // 1. Padded LineBox wrapping a 17px-tall hollow-loop cursive word (y=50..66 inside box y=36..80)
    const w1 = 380;
    const h1 = 120;
    const binHollow = new Uint8ClampedArray(w1 * h1).fill(255);
    for (let x = 20; x <= 360; x++) {
      // Top & bottom arcs of hollow loops (y=50..51 and y=65..66) plus vertical loop sides every 18px
      for (const y of [50, 51, 65, 66]) {
        binHollow[y * w1 + x] = 0;
      }
      if ((x - 20) % 18 < 2) {
        for (let y = 50; y <= 66; y++) binHollow[y * w1 + x] = 0;
      }
    }
    const hollowChunks = splitWideLineBox(
      binHollow,
      w1,
      h1,
      { x0: 20, y0: 36, x1: 360, y1: 80 },
      6.5
    );
    expect(hollowChunks.length).toBe(1);

    // 2. 2px-wide ligature bridge between tall stems (x=145..146)
    const w2 = 380;
    const h2 = 60;
    const binNarrowLig = new Uint8ClampedArray(w2 * h2).fill(255);
    for (let y = 12; y <= 45; y++) {
      for (let x = 10; x <= 144; x++) binNarrowLig[y * w2 + x] = 0;
      for (let x = 147; x <= 340; x++) binNarrowLig[y * w2 + x] = 0;
    }
    for (let y = 34; y <= 35; y++) {
      binNarrowLig[y * w2 + 145] = 0;
      binNarrowLig[y * w2 + 146] = 0;
    }
    const narrowLigChunks = splitWideLineBox(
      binNarrowLig,
      w2,
      h2,
      { x0: 10, y0: 12, x1: 340, y1: 45 },
      5.0
    );
    expect(narrowLigChunks.length).toBe(2);
    expect(narrowLigChunks[0].x1).toBeGreaterThanOrEqual(145);
    expect(narrowLigChunks[0].x1).toBeLessThanOrEqual(146);
  });
});
