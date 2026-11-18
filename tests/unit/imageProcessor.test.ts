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
});
