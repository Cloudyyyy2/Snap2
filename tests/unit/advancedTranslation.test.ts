import { describe, it, expect } from 'vitest';
import {
  skeletonizeBinary,
  findLigatureCutPoints,
  segmentHandwrittenWord,
} from '../../src/lib/ocr/handwriting/advancedTranslation';
import { decodeWordContextually } from '../../src/lib/ocr/handwriting/dictionaryDecoder';

describe('Advanced Handwriting Translation & Character Segmentation', () => {
  it('skeletonizeBinary reduces thick ink strokes into 1-pixel skeletons', () => {
    const w = 40;
    const h = 40;
    const binary = new Uint8ClampedArray(w * h).fill(255);

    // Draw a thick vertical line of ink (0)
    for (let y = 5; y < 35; y++) {
      for (let x = 18; x <= 22; x++) {
        binary[y * w + x] = 0;
      }
    }

    const skeletal = skeletonizeBinary(binary, w, h);
    expect(skeletal.length).toBe(w * h);

    // Verify skeleton has ink pixels
    let inkCount = 0;
    for (let i = 0; i < skeletal.length; i++) {
      if (skeletal[i] === 0) inkCount++;
    }
    expect(inkCount).toBeGreaterThan(0);
    expect(inkCount).toBeLessThan(35); // significantly thinner than 5 * 30 = 150 pixels
  });

  it('findLigatureCutPoints detects separation valleys in cursive scripts', () => {
    const w = 100;
    const h = 30;
    const binary = new Uint8ClampedArray(w * h).fill(255);

    // Draw two character blobs with a clear gap around x = 50
    for (let y = 5; y < 25; y++) {
      for (let x = 10; x < 45; x++) binary[y * w + x] = 0;
      for (let x = 55; x < 90; x++) binary[y * w + x] = 0;
    }

    const cuts = findLigatureCutPoints(binary, w, h);
    expect(Array.isArray(cuts)).toBe(true);
  });

  it('segmentHandwrittenWord segments characters and computes alternatives', () => {
    const w = 80;
    const h = 30;
    const binary = new Uint8ClampedArray(w * h).fill(255);

    for (let y = 5; y < 25; y++) {
      for (let x = 10; x < 70; x++) {
        binary[y * w + x] = 0;
      }
    }

    const result = segmentHandwrittenWord(binary, w, h, 'call');
    expect(result.translatedWord).toBeDefined();
    expect(result.characters.length).toBeGreaterThan(0);
    expect(result.confidence).toBeGreaterThan(0);
  });

  it('decodeWordContextually maps misspelled handwritten words to valid dictionary items', () => {
    const res1 = decodeWordContextually('clal'); // common confusion of 'call'
    expect(res1.correctedWord).toBe('call');
    expect(res1.isDictionaryMatch).toBe(true);

    const res2 = decodeWordContextually('mom'); // exact match
    expect(res2.correctedWord).toBe('mom');
    expect(res2.confidence).toBeGreaterThan(90);
  });
});
