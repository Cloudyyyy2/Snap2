import { describe, it, expect } from 'vitest';
import {
  isPointInPolygon,
  isBoxInPolygon,
  detectTextBlockQuad,
  trackLiveTextLines,
} from '../../src/lib/ocr/documentEdgeDetector';
import { flattenDocumentLighting } from '../../src/lib/ocr/preprocess';

describe('Modern Crop Enhancement & Text Containment Pipeline', () => {
  describe('Ray-Casting Point and Box Polygon Containment', () => {
    // Unit square polygon in normalized coordinates
    const quad = [
      { x: 0.1, y: 0.1 },
      { x: 0.9, y: 0.1 },
      { x: 0.9, y: 0.9 },
      { x: 0.1, y: 0.9 },
    ];

    it('correctly classifies points inside vs outside the quad', () => {
      // Point at center
      expect(isPointInPolygon({ x: 0.5, y: 0.5 }, quad)).toBe(true);

      // Points well outside
      expect(isPointInPolygon({ x: 0.02, y: 0.5 }, quad)).toBe(false);
      expect(isPointInPolygon({ x: 0.98, y: 0.5 }, quad)).toBe(false);
      expect(isPointInPolygon({ x: 0.5, y: 0.02 }, quad)).toBe(false);
      expect(isPointInPolygon({ x: 0.5, y: 0.98 }, quad)).toBe(false);
    });

    it('correctly handles an arbitrary angled trapezoidal quad', () => {
      // Perspective trapezoid: wider at bottom than top
      const trapezoid = [
        { x: 0.25, y: 0.1 },
        { x: 0.75, y: 0.1 },
        { x: 0.95, y: 0.85 },
        { x: 0.05, y: 0.85 },
      ];

      expect(isPointInPolygon({ x: 0.5, y: 0.5 }, trapezoid)).toBe(true);
      // Top corner outside the narrow trapezoid top
      expect(isPointInPolygon({ x: 0.15, y: 0.12 }, trapezoid)).toBe(false);
      // Bottom corner inside the wide trapezoid bottom
      expect(isPointInPolygon({ x: 0.15, y: 0.8 }, trapezoid)).toBe(true);
    });

    it('isBoxInPolygon identifies fully enclosed text boxes', () => {
      const insideBox = {
        x: 0.2,
        y: 0.2,
        width: 0.4,
        height: 0.1,
      };
      expect(isBoxInPolygon(insideBox, quad)).toBe(true);
    });

    it('isBoxInPolygon flags partially clipped or fully exterior text boxes', () => {
      // Partially clipped over the left edge
      const clippedBox = {
        x: 0.05,
        y: 0.3,
        width: 0.2,
        height: 0.1,
      };
      expect(isBoxInPolygon(clippedBox, quad)).toBe(false);

      // Entirely outside the bottom edge
      const outsideBox = {
        x: 0.3,
        y: 0.95,
        width: 0.4,
        height: 0.04,
      };
      expect(isBoxInPolygon(outsideBox, quad)).toBe(false);
    });
  });

  describe('flattenDocumentLighting (Retinex illumination & shadow removal)', () => {
    it('flattens shadow gradients while boosting ink contrast on RGBA images', () => {
      const w = 120;
      const h = 80;
      const rgba = new Uint8ClampedArray(w * h * 4);

      // Create a severe diagonal lighting gradient (simulating room / desk shadow)
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const off = (y * w + x) * 4;
          // Gradient from dark shadow (100) on top-left to bright light (220) on bottom-right
          const bgVal = Math.round(100 + ((x + y) / (w + h)) * 120);

          rgba[off] = bgVal;
          rgba[off + 1] = bgVal;
          rgba[off + 2] = bgVal;
          rgba[off + 3] = 255;
        }
      }

      // Draw horizontal ink strokes (dark lines = 30) across both shadow and bright zones
      for (let y = 25; y < 30; y++) {
        for (let x = 15; x < 105; x++) {
          const off = (y * w + x) * 4;
          rgba[off] = 30;
          rgba[off + 1] = 30;
          rgba[off + 2] = 30;
          rgba[off + 3] = 255;
        }
      }

      const flattened = flattenDocumentLighting(rgba, w, h);
      expect(flattened.length).toBe(w * h * 4);

      // Check paper regions in both shadowed and bright zones: both should be lifted towards clean white
      const shadowPaperIdx = (10 * w + 10) * 4;
      const brightPaperIdx = (70 * w + 100) * 4;
      expect(flattened[shadowPaperIdx]).toBeGreaterThan(200);
      expect(flattened[brightPaperIdx]).toBeGreaterThan(200);

      // Check ink stroke: remains dark with crisp high contrast
      const inkIdx = (27 * w + 50) * 4;
      expect(flattened[inkIdx]).toBeLessThan(70);

      // Contrast between paper and ink should exceed 150 points
      expect(flattened[shadowPaperIdx] - flattened[inkIdx]).toBeGreaterThan(150);
    });
  });

  describe('detectTextBlockQuad', () => {
    it('extracts tight bounding quad around text block ink', () => {
      const w = 200;
      const h = 150;
      const gray = new Uint8ClampedArray(w * h).fill(240); // bright paper

      // Draw a centered paragraph of text strokes
      for (let y = 40; y < 100; y += 12) {
        for (let dy = 0; dy < 3; dy++) {
          const rowOff = (y + dy) * w;
          for (let x = 30; x < 170; x++) {
            if (x % 14 >= 11) continue; // word spaces
            gray[rowOff + x] = 25; // dark ink
          }
        }
      }

      const quad = detectTextBlockQuad(gray, w, h);
      expect(quad).not.toBeNull();
      if (quad) {
        // Enclosing quad should contain the text block (minX ~ 30, maxX ~ 170, minY ~ 40, maxY ~ 100)
        expect(quad[0].x).toBeLessThanOrEqual(30);
        expect(quad[0].y).toBeLessThanOrEqual(40);
        expect(quad[1].x).toBeGreaterThanOrEqual(160);
        expect(quad[2].y).toBeGreaterThanOrEqual(95);
      }
    });

    it('returns null on blank paper or uniform images with no text', () => {
      const w = 100;
      const h = 100;
      const blankGray = new Uint8ClampedArray(w * h).fill(245);
      const quad = detectTextBlockQuad(blankGray, w, h);
      expect(quad).toBeNull();
    });
  });

  describe('trackLiveTextLines (Live Viewfinder Text Detection)', () => {
    function createMockCanvas(
      width: number,
      height: number,
      drawLines: Array<{ y: number; height: number; x0: number; x1: number }>
    ) {
      const data = new Uint8ClampedArray(width * height * 4).fill(255);
      for (const line of drawLines) {
        for (let y = line.y; y < line.y + line.height; y++) {
          if (y < 0 || y >= height) continue;
          for (let x = line.x0; x < line.x1; x++) {
            if (x < 0 || x >= width) continue;
            const idx = (y * width + x) * 4;
            data[idx] = 20;
            data[idx + 1] = 20;
            data[idx + 2] = 20;
            data[idx + 3] = 255;
          }
        }
      }
      return {
        width,
        height,
        getContext: () => ({
          getImageData: (_x: number, _y: number, w: number, h: number) => ({
            width: w,
            height: h,
            data,
          }),
        }),
      } as unknown as HTMLCanvasElement;
    }

    it('returns empty results on blank or undersized frames', () => {
      const smallCanvas = createMockCanvas(30, 30, []);
      const result = trackLiveTextLines(smallCanvas);
      expect(result.isTextInFrame).toBe(false);
      expect(result.lineCount).toBe(0);

      const blankCanvas = createMockCanvas(200, 150, []);
      const blankResult = trackLiveTextLines(blankCanvas);
      expect(blankResult.isTextInFrame).toBe(false);
      expect(blankResult.lines.length).toBe(0);
    });

    it('tracks live text lines and computes smoothed bounding ribbons and text hull', () => {
      const w = 240;
      const h = 180;
      // 2 horizontal text lines
      const mockCanvas = createMockCanvas(w, h, [
        { y: 40, height: 14, x0: 30, x1: 200 },
        { y: 80, height: 14, x0: 30, x1: 180 },
      ]);

      const res = trackLiveTextLines(mockCanvas);
      expect(res.isTextInFrame).toBe(true);
      expect(res.lineCount).toBeGreaterThanOrEqual(1);
      expect(res.textHullBox).toBeDefined();
      if (res.textHullBox) {
        expect(res.textHullBox.width).toBeGreaterThan(0.4);
        expect(res.textHullBox.height).toBeGreaterThan(0.2);
      }

      // Test EMA smoothing with previous frame lines
      const resSmoothed = trackLiveTextLines(mockCanvas, res.lines);
      expect(resSmoothed.lines.length).toBeGreaterThanOrEqual(1);
      expect(resSmoothed.lines[0].id).toBe(res.lines[0].id);
    });
  });
});

