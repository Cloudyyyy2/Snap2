import { describe, it, expect } from 'vitest';
import {
  rgbaToGrayscale,
  rgbToGrayscale,
  toGrayscale,
  grayscaleToRgba,
  applyMedianFilter3x3,
  medianFilter,
  applyGaussianBlur,
  gaussianBlur,
  estimateNoiseLevel,
  binarizeSauvola,
  sauvolaBinarize,
  preprocessForOcr,
  preprocessImageData,
  twoPassContourVoting,
} from '../../src/lib/ocr/preprocess';

describe('src/lib/ocr/preprocess OCR image preprocessing utility', () => {
  it('converts RGBA, RGB, and transparent pixels accurately to ITU-R BT.601 grayscale', () => {
    const w = 4;
    const h = 1;
    // Pixel 0: Pure Red (77/256 ≈ 76), Pixel 1: Pure Green (150/256 ≈ 150),
    // Pixel 2: Transparent black (alpha=0 -> composites onto white 255),
    // Pixel 3: Dark ink RGB(20, 25, 30)
    const rgba = new Uint8ClampedArray([
      255, 0, 0, 255,
      0, 255, 0, 255,
      0, 0, 0, 0,
      20, 25, 30, 255,
    ]);

    const gray = rgbaToGrayscale(rgba, w, h);
    expect(gray[0]).toBeGreaterThanOrEqual(74);
    expect(gray[0]).toBeLessThanOrEqual(78);
    expect(gray[1]).toBeGreaterThanOrEqual(148);
    expect(gray[1]).toBeLessThanOrEqual(152);
    expect(gray[2]).toBe(255);
    expect(gray[3]).toBeLessThan(30);

    const rgb = new Uint8ClampedArray([255, 255, 255, 10, 10, 10]);
    const rgbGray = rgbToGrayscale(rgb, 2, 1);
    expect(rgbGray[0]).toBe(255);
    expect(rgbGray[1]).toBe(10);

    // Auto-detection via toGrayscale and round-trip RGBA helper
    const autoGray = toGrayscale(rgba, w, h);
    expect(autoGray.length).toBe(4);
    expect(autoGray[2]).toBe(255);

    const roundTrip = grayscaleToRgba(rgbGray, 2, 1);
    expect(Array.from(roundTrip)).toEqual([255, 255, 255, 255, 10, 10, 10, 255]);
  });

  it('applyMedianFilter3x3 strips salt-and-pepper noise while preserving crisp ink edges', () => {
    const w = 60;
    const h = 40;
    const buf = new Uint8ClampedArray(w * h).fill(245);

    // Solid dark printed stroke bar
    for (let y = 14; y <= 24; y++) {
      for (let x = 15; x <= 45; x++) {
        buf[y * w + x] = 20;
      }
    }
    // Inject isolated salt & pepper artifacts
    buf[5 * w + 5] = 0;
    buf[6 * w + 48] = 10;
    buf[19 * w + 30] = 255;

    const clean = applyMedianFilter3x3(buf, w, h);
    expect(clean[5 * w + 5]).toBe(245);
    expect(clean[6 * w + 48]).toBe(245);
    expect(clean[19 * w + 30]).toBe(20);
    // Stroke boundary remains sharp
    expect(clean[14 * w + 25]).toBe(20);
    expect(clean[13 * w + 25]).toBe(245);
  });

  it('applyGaussianBlur smooths high-frequency sensor grain while preserving mean brightness', () => {
    const w = 48;
    const h = 48;
    const grainy = new Uint8ClampedArray(w * h);

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        // High-frequency +/- 24 checkerboard grain around mean 210
        grainy[y * w + x] = (x + y) % 2 === 0 ? 234 : 186;
      }
    }

    const smoothed = applyGaussianBlur(grainy, w, h, 1.1);
    let minVal = 255;
    let maxVal = 0;
    let sum = 0;
    for (let y = 4; y < h - 4; y++) {
      for (let x = 4; x < w - 4; x++) {
        const v = smoothed[y * w + x];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
        sum += v;
      }
    }

    const mean = sum / ((w - 8) * (h - 8));
    expect(mean).toBeGreaterThan(206);
    expect(mean).toBeLessThan(214);
    expect(maxVal - minVal).toBeLessThan(8);
  });

  it('binarizeSauvola isolates text cleanly across steep uneven lighting shadows', () => {
    const w = 160;
    const h = 100;
    const shaded = new Uint8ClampedArray(w * h);

    // Paper brightness drops steeply from 245 (top-left) to 110 (bottom-right)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        shaded[y * w + x] = Math.round(245 - (x / w) * 75 - (y / h) * 60);
      }
    }

    // Ink strokes in bright top-left (bg~235, ink~145) AND dark bottom-right (bg~125, ink~40)
    for (let y = 16; y <= 26; y++) {
      for (let x = 20; x <= 60; x++) shaded[y * w + x] = 145;
    }
    for (let y = 72; y <= 82; y++) {
      for (let x = 100; x <= 145; x++) shaded[y * w + x] = 40;
    }

    const bin = sauvolaBinarize(shaded, w, h, 14, 0.2);
    expect(bin).toEqual(binarizeSauvola(shaded, w, h, 14, 0.2));
    expect(medianFilter).toBe(applyMedianFilter3x3);
    expect(gaussianBlur).toBe(applyGaussianBlur);
    // Both bright-zone and shadowed-zone strokes detected as ink (0)
    expect(bin[21 * w + 40]).toBe(0);
    expect(bin[77 * w + 122]).toBe(0);
    // Nearby shaded background stays clean white (255)
    expect(bin[60 * w + 122]).toBe(255);
    expect(bin[40 * w + 40]).toBe(255);
  });

  it('preprocessForOcr and preprocessImageData normalize dark-mode UI, uneven paper, noise, and tilted text end-to-end', () => {
    const w = 220;
    const h = 140;
    const rgba = new Uint8ClampedArray(w * h * 4);

    // 1. Noisy, shadowed light-paper RGBA handwriting sample
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const bg = Math.round(240 - (x / w) * 60);
        const idx = (y * w + x) * 4;
        rgba[idx] = bg;
        rgba[idx + 1] = bg;
        rgba[idx + 2] = bg;
        rgba[idx + 3] = 255;
      }
    }

    // Tilted (+5 deg) text stroke bar across middle
    const rad = (5 * Math.PI) / 180;
    const tan = Math.tan(rad);
    for (let x = 30; x < 190; x++) {
      if (x % 20 >= 15) continue;
      const cy = 70 - (x - w / 2) * tan;
      for (let dy = -4; dy <= 4; dy++) {
        const py = Math.round(cy + dy);
        if (py >= 0 && py < h) {
          const idx = (py * w + x) * 4;
          rgba[idx] = 28;
          rgba[idx + 1] = 30;
          rgba[idx + 2] = 35;
        }
      }
    }
    // Salt & pepper speck
    rgba[(10 * w + 10) * 4] = 0;
    rgba[(10 * w + 10) * 4 + 1] = 0;
    rgba[(10 * w + 10) * 4 + 2] = 0;

    const result = preprocessImageData(
      { data: rgba, width: w, height: h },
      {
        profile: 'handwriting',
        denoise: 'median+gaussian',
        gaussianSigma: 0.75,
        binarize: 'sauvola',
        deskew: true,
      }
    );

    expect(result.width).toBe(w);
    expect(result.height).toBe(h);
    expect(result.inverted).toBe(false);
    expect(result.appliedSteps).toContain('grayscale-luminance');
    expect(result.appliedSteps).toContain('illumination-flatten');
    expect(result.appliedSteps).toContain('median-3x3');
    expect(result.appliedSteps).toContain('sauvola-binarize');
    expect(result.appliedSteps).toContain('ink-contrast-stretch');
    expect(result.skewAngle).toBe(0);

    // Speck removed cleanly
    expect(result.binary[10 * w + 10]).toBe(255);
    // Leveled center stroke preserved cleanly as foreground ink
    expect(result.binary[70 * w + 108]).toBe(0);

    // 2. Dark-mode RGBA frame auto-inverts cleanly
    const darkRgba = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      const idx = i * 4;
      darkRgba[idx] = 18;
      darkRgba[idx + 1] = 20;
      darkRgba[idx + 2] = 26;
      darkRgba[idx + 3] = 255;
    }
    for (let y = 55; y <= 75; y++) {
      for (let x = 40; x <= 180; x++) {
        const idx = (y * w + x) * 4;
        darkRgba[idx] = 238;
        darkRgba[idx + 1] = 240;
        darkRgba[idx + 2] = 245;
      }
    }

    const darkRes = preprocessForOcr(darkRgba, w, h, {
      profile: 'printed',
      denoise: 'median+bilateral',
      binarize: 'hybrid',
    });
    expect(darkRes.inverted).toBe(true);
    expect(darkRes.appliedSteps).toContain('color-inversion');
    expect(darkRes.binary[65 * w + 100]).toBe(0);
    expect(darkRes.binary[10 * w + 10]).toBe(255);
  });

  it('estimateNoiseLevel distinguishes clean screenshots from noisy photos and triggers adaptive Gaussian denoising', () => {
    const w = 160;
    const h = 100;
    const clean = new Uint8ClampedArray(w * h).fill(248);
    for (let y = 40; y < 55; y++) {
      for (let x = 30; x < 130; x++) clean[y * w + x] = 25;
    }
    expect(estimateNoiseLevel(clean, w, h)).toBe(0);

    const noisy = new Uint8ClampedArray(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        noisy[y * w + x] = (x + y) % 2 === 0 ? 250 : 210;
      }
    }
    expect(estimateNoiseLevel(noisy, w, h)).toBeGreaterThanOrEqual(4.0);

    const autoNoisyRes = preprocessForOcr(noisy, w, h, { denoise: 'auto' });
    expect(autoNoisyRes.appliedSteps.some((s) => s.startsWith('gaussian('))).toBe(true);
  });

  it('twoPassContourVoting recovers faint ink lines while suppressing isolated speckles', () => {
    const w = 120;
    const h = 80;
    const gray = new Uint8ClampedArray(w * h).fill(240); // Paper white

    // Draw faint pencil stroke (gray 130) across row 40
    for (let x = 20; x < 100; x++) {
      gray[40 * w + x] = 128;
    }
    // Random isolated single-pixel salt noise speckle (gray 60)
    gray[15 * w + 15] = 60;

    const res = twoPassContourVoting(gray, w, h);
    expect(res.binary).toBeInstanceOf(Uint8ClampedArray);
    expect(res.binary.length).toBe(w * h);
    expect(res.inkRatio).toBeGreaterThan(0.005);
    // The faint line at row 40 should be preserved as black ink (0)
    expect(res.binary[40 * w + 50]).toBe(0);
    // Isolated speckle at (15, 15) should be suppressed or paper (255)
    expect(res.binary[15 * w + 15]).toBe(255);
  });
});
