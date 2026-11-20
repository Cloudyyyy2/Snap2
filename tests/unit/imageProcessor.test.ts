import { describe, it, expect } from 'vitest';
import { preprocessForOCR } from '../../src/services/imageProcessor';

describe('Image Processor Service Utility - preprocessForOCR', () => {
  it('processes HTMLCanvasElement and returns enhanced ImageData using adaptive thresholding', () => {
    const w = 100;
    const h = 80;
    const rgba = new Uint8ClampedArray(w * h * 4).fill(255);

    for (let y = 35; y < 45; y++) {
      for (let x = 15; x < 85; x++) {
        const idx = (y * w + x) * 4;
        rgba[idx] = 20;
        rgba[idx + 1] = 20;
        rgba[idx + 2] = 20;
        rgba[idx + 3] = 255;
      }
    }

    const mockCanvas = {
      width: w,
      height: h,
      getContext: () => ({
        getImageData: () => ({
          width: w,
          height: h,
          data: rgba,
        }),
      }),
    } as unknown as HTMLCanvasElement;

    const result = preprocessForOCR(mockCanvas);
    expect(result).toBeDefined();
    expect(result.width).toBe(w);
    expect(result.height).toBe(h);
    expect(result.data).toBeDefined();
    expect(result.data.length).toBe(w * h * 4);
  });

  it('processes ImageData directly and returns enhanced ImageData', () => {
    const w = 60;
    const h = 50;
    const data = new Uint8ClampedArray(w * h * 4).fill(255);
    const imgData = { width: w, height: h, data };

    const result = preprocessForOCR(imgData as ImageData);
    expect(result).toBeDefined();
    expect(result.width).toBe(w);
    expect(result.height).toBe(h);
    expect(result.data).toBeDefined();
  });

  it('preserves continuous anti-aliased grayscale strokes (0 < v < 255) by default and supports explicit binary mode', () => {
    const w = 120;
    const h = 80;
    const rgba = new Uint8ClampedArray(w * h * 4).fill(255);
    // Paper background = 245
    for (let i = 0; i < w * h; i++) {
      const idx = i * 4;
      rgba[idx] = 245;
      rgba[idx + 1] = 245;
      rgba[idx + 2] = 245;
      rgba[idx + 3] = 255;
    }

    // Anti-aliased horizontal stroke with dark core (20) and gray fringe rows (110, 165)
    for (let x = 20; x < 100; x++) {
      const setPx = (y: number, v: number) => {
        const idx = (y * w + x) * 4;
        rgba[idx] = v;
        rgba[idx + 1] = v;
        rgba[idx + 2] = v;
      };
      setPx(37, 165);
      setPx(38, 110);
      setPx(39, 20);
      setPx(40, 20);
      setPx(41, 110);
      setPx(42, 165);
    }

    const imgData = { width: w, height: h, data: rgba } as ImageData;

    const grayResult = preprocessForOCR(imgData);
    const explicitGrayResult = preprocessForOCR(imgData, { mode: 'grayscale' });

    let intermediateCount = 0;
    for (let i = 0; i < w * h; i++) {
      const idx = i * 4;
      const r = grayResult.data[idx];
      const g = grayResult.data[idx + 1];
      const b = grayResult.data[idx + 2];
      const a = grayResult.data[idx + 3];
      expect(r).toBe(g);
      expect(g).toBe(b);
      expect(a).toBe(255);
      expect(explicitGrayResult.data[idx]).toBe(r);
      if (r > 0 && r < 255) {
        intermediateCount++;
      }
    }
    expect(intermediateCount).toBeGreaterThan(20);

    const binaryResult = preprocessForOCR(imgData, { mode: 'binary' });
    for (let i = 0; i < w * h; i++) {
      const v = binaryResult.data[i * 4];
      expect(v === 0 || v === 255).toBe(true);
    }
  });
});
