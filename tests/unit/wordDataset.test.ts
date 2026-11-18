import { describe, it, expect } from 'vitest';
import {
  damerauLevenshteinDistance,
  correctWordWithLexicon,
  correctTextWithLexicon,
  ENGLISH_REMINDER_WORDS,
  LEXICON_SET,
} from '../../src/lib/ocr/wordDataset';

describe('built-in English and reminder vocabulary dataset & fuzzy spell corrector', () => {
  it('contains essential reminder, task, scheduling, and action words', () => {
    expect(LEXICON_SET.has('dentist')).toBe(true);
    expect(LEXICON_SET.has('groceries')).toBe(true);
    expect(LEXICON_SET.has('tomorrow')).toBe(true);
    expect(LEXICON_SET.has('appointment')).toBe(true);
    expect(LEXICON_SET.has('doctor')).toBe(true);
    expect(LEXICON_SET.has('electric')).toBe(true);
    expect(LEXICON_SET.has('bill')).toBe(true);
    expect(ENGLISH_REMINDER_WORDS.length).toBeGreaterThan(150);
  });

  it('damerauLevenshteinDistance computes accurate edit distances including adjacent transpositions', () => {
    expect(damerauLevenshteinDistance('dentist', 'dentist')).toBe(0);
    expect(damerauLevenshteinDistance('dentst', 'dentist')).toBe(1); // insertion
    expect(damerauLevenshteinDistance('dentistt', 'dentist')).toBe(1); // deletion
    expect(damerauLevenshteinDistance('dentyst', 'dentist')).toBe(1); // substitution
    expect(damerauLevenshteinDistance('dnetist', 'dentist')).toBe(1); // transposition 'en' <-> 'ne'
  });

  it('correctWordWithLexicon repairs OCR handwriting confusion substitutions', () => {
    // Optical confusion 'rn' mistaken for 'm'
    const rnRes = correctWordWithLexicon('rnilk');
    expect(rnRes.wasCorrected).toBe(true);
    expect(rnRes.corrected.toLowerCase()).toBe('milk');

    // 'cl' mistaken for 'd'
    const clRes = correctWordWithLexicon('clate');
    expect(clRes.wasCorrected).toBe(true);
    expect(clRes.corrected.toLowerCase()).toBe('date');

    // Fuzzy match within distance 2
    const docRes = correctWordWithLexicon('doctar');
    expect(docRes.wasCorrected).toBe(true);
    expect(docRes.corrected.toLowerCase()).toBe('doctor');

    // 'grocenes' handwriting misread -> 'groceries'
    const grocRes = correctWordWithLexicon('grocenes');
    expect(grocRes.wasCorrected).toBe(true);
    expect(grocRes.corrected.toLowerCase()).toBe('groceries');

    // 'tommorow' / 'tomorow' typo -> 'tomorrow'
    const tomRes = correctWordWithLexicon('tommorow');
    expect(tomRes.wasCorrected).toBe(true);
    expect(tomRes.corrected.toLowerCase()).toBe('tomorrow');
  });

  it('correctWordWithLexicon preserves dates, numbers, currency, and times verbatim', () => {
    expect(correctWordWithLexicon('3pm').corrected).toBe('3pm');
    expect(correctWordWithLexicon('10:30am').corrected).toBe('10:30am');
    expect(correctWordWithLexicon('$45.50').corrected).toBe('$45.50');
    expect(correctWordWithLexicon('2026').corrected).toBe('2026');
  });

  it('correctTextWithLexicon restores handwritten reminder text and delivers 90%+ accuracy score', () => {
    const rawOcr = 'buy grocenes tommorow at 3pm\ncall dentst for appointment';
    const result = correctTextWithLexicon(rawOcr);

    expect(result.correctedText).toContain('groceries');
    expect(result.correctedText).toContain('tomorrow');
    expect(result.correctedText).toContain('dentist');
    expect(result.correctedText).toContain('appointment');
    expect(result.corrections.length).toBeGreaterThanOrEqual(2);
    expect(result.accuracyScore).toBeGreaterThanOrEqual(90);
    expect(result.wordsVerified).toBeGreaterThanOrEqual(result.totalWords - 1);
  });
});
