import { describe, it, expect } from 'vitest';
import {
  hasErraticTesseractArtifacts,
  scoreTextPlausibility,
} from '../../src/lib/ocr/plausibility';

describe('OCR plausibility and erratic Tesseract artifact detection', () => {
  it('detects erratic Tesseract artifacts in garbled handwriting misreads', () => {
    const garbled = '0/1, Print expert and assessment "foo" "bar" 0,000551';
    expect(hasErraticTesseractArtifacts(garbled)).toBe(true);
    expect(hasErraticTesseractArtifacts('')).toBe(true);
    expect(
      hasErraticTesseractArtifacts('bread milk eggs', [{ confidence: 0.58 }])
    ).toBe(true);
  });

  it('does not flag clean grocery or reminder lines as erratic', () => {
    const clean = 'bread\nmilk\nCrisps\neggs\nPain au chocolat';
    expect(
      hasErraticTesseractArtifacts(clean, [
        { confidence: 0.92 },
        { confidence: 0.9 },
      ])
    ).toBe(false);
  });

  it('scores clean grocery lines higher than erratic Tesseract output', () => {
    const clean = 'bread\nmilk\nCrisps\neggs\nasparagus\nturkey';
    const garbled = '0/1, Print expert and assessment "foo" "bar" 0,000551';

    const cleanScore = scoreTextPlausibility(clean, 0.9, [{ confidence: 0.9 }]);
    const garbledScore = scoreTextPlausibility(garbled, 0.55, [{ confidence: 0.55 }]);

    expect(cleanScore).toBeGreaterThan(garbledScore);
    expect(cleanScore).toBeGreaterThan(0.75);
    expect(garbledScore).toBeLessThan(0.5);
    expect(scoreTextPlausibility('', 0.9)).toBe(0);
  });

  it('flags leading-zero-comma noise without line confidences and scores it below 0.68', () => {
    const garbledNoLines = '0/1, Print expert and assessment 0,000551';
    expect(hasErraticTesseractArtifacts(garbledNoLines)).toBe(true);
    expect(scoreTextPlausibility(garbledNoLines, 0.7)).toBeLessThan(0.68);
  });

  it('does not false-positive valid recipe fractions, numbered lists, or paired quotes', () => {
    expect(hasErraticTesseractArtifacts('1/2 cup milk\n3/4 cup sugar')).toBe(false);
    expect(hasErraticTesseractArtifacts('1 milk\n2 eggs\n3 bread')).toBe(false);
    expect(hasErraticTesseractArtifacts('Read "Atomic Habits" tonight')).toBe(false);
  });
});
