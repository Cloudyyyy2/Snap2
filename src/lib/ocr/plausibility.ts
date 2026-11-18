import { getCompactLexicon } from './compactLexicon';

const CULINARY_FRACTION_RE = /^(?:1\/2|1\/3|2\/3|1\/4|3\/4|1\/8|3\/8|5\/8|7\/8),?$/;
const LEADING_ZERO_COMMA_NOISE_RE = /^0,\d{3,}$/;
const ZERO_SLASH_NOISE_RE = /^0\/\d*,?$/;

function getNumberedListLineIndices(text: string): Set<number> {
  const validIndices = new Set<number>();
  const lines = text.split(/\r?\n/);
  let tokenOffset = 0;
  for (const line of lines) {
    const lineTokens = line.trim().split(/\s+/).filter(Boolean);
    if (
      lineTokens.length >= 2 &&
      /^[1-9][.)]?$/.test(lineTokens[0]) &&
      /[A-Za-z]{2,}/.test(lineTokens[1])
    ) {
      validIndices.add(tokenOffset);
    }
    tokenOffset += lineTokens.length;
  }
  return validIndices;
}

export function hasErraticTesseractArtifacts(
  text: string,
  lines?: Array<{ confidence: number }>
): boolean {
  const trimmed = text.trim();
  if (!trimmed) return true;

  if (lines && lines.some((l) => l.confidence < 0.72)) {
    return true;
  }

  const tokens = trimmed.split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return true;

  const listPrefixIndices = getNumberedListLineIndices(trimmed);
  let strayDigitOrQuoteCount = 0;
  let hasSevereZeroNoise = false;

  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    if (LEADING_ZERO_COMMA_NOISE_RE.test(tok) || /^0{2,}[,.]?0*\d*$/.test(tok)) {
      hasSevereZeroNoise = true;
      strayDigitOrQuoteCount++;
      continue;
    }
    if (ZERO_SLASH_NOISE_RE.test(tok)) {
      strayDigitOrQuoteCount++;
      continue;
    }
    if (/^[0-9]$/.test(tok)) {
      if (!listPrefixIndices.has(i)) {
        strayDigitOrQuoteCount++;
      }
      continue;
    }
    if (/^["'“”`]+$/.test(tok)) {
      strayDigitOrQuoteCount++;
      continue;
    }
    if (/^\d+\/\d*,?$/.test(tok) && !CULINARY_FRACTION_RE.test(tok)) {
      // Allow valid dates like 5/12, flag malformed or 0-numerator slashes
      if (/^\d+\/$/.test(tok) || /^0/.test(tok)) {
        strayDigitOrQuoteCount++;
      }
    }
  }

  if (hasSevereZeroNoise) {
    return true;
  }

  if (strayDigitOrQuoteCount >= 2 && strayDigitOrQuoteCount / tokens.length >= 0.14) {
    return true;
  }

  // Repeated zero groups like "0 0000,000" or unbalanced/excessive stray quotes
  const quoteMatches = trimmed.match(/"/g) || [];
  const hasUnbalancedQuotes = quoteMatches.length % 2 !== 0;
  const hasExcessiveQuotes = quoteMatches.length >= 4;
  if (/\b0\s+0{3,}[,.]?0*\b/.test(trimmed) || hasUnbalancedQuotes || hasExcessiveQuotes) {
    return true;
  }

  return false;
}

export function scoreTextPlausibility(
  text: string,
  qualityScore: number,
  lines?: Array<{ confidence: number }>
): number {
  const trimmed = text.trim();
  if (!trimmed) return 0;

  const rawTokens = trimmed.split(/\s+/).filter(Boolean);
  if (rawTokens.length === 0) return 0;

  const lexicon = getCompactLexicon();
  const listPrefixIndices = getNumberedListLineIndices(trimmed);
  let lexiconOrNumericHits = 0;
  let alphaNumericTokens = 0;
  let weirdSymbolTokens = 0;
  let strayNoiseTokens = 0;

  for (let i = 0; i < rawTokens.length; i++) {
    const rawTok = rawTokens[i];
    const cleaned = rawTok.toLowerCase().replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, '');
    if (!cleaned) {
      weirdSymbolTokens++;
      continue;
    }

    const isLeadingZeroNoise =
      LEADING_ZERO_COMMA_NOISE_RE.test(cleaned) || /^0{2,}/.test(cleaned);
    const isZeroSlashNoise = ZERO_SLASH_NOISE_RE.test(rawTok);
    const isStraySingleDigit = /^[0-9]$/.test(rawTok) && !listPrefixIndices.has(i);
    const isIsolatedQuote = /^["']+$/.test(rawTok);

    if (isLeadingZeroNoise || isZeroSlashNoise || isStraySingleDigit || isIsolatedQuote) {
      strayNoiseTokens++;
    }
    alphaNumericTokens++;

    if (isLeadingZeroNoise || isZeroSlashNoise) {
      continue;
    }

    if (
      lexicon.has(cleaned) ||
      /^\d+(?:[.,:/-]\d+)*(?:am|pm|k|m)?$/.test(cleaned) ||
      (/^[a-z]{3,}$/.test(cleaned) && /[aeiouy]/.test(cleaned))
    ) {
      lexiconOrNumericHits++;
    }
  }

  if (alphaNumericTokens === 0) return 0.05;

  const lexRatio = lexiconOrNumericHits / rawTokens.length;
  const cleanRatio = Math.max(0, 1 - weirdSymbolTokens / rawTokens.length);
  const noisePenalty = Math.min(0.4, (strayNoiseTokens / rawTokens.length) * 0.95);
  const lowConfPenalty =
    lines && lines.length > 0 && lines.some((l) => l.confidence < 0.7) ? 0.12 : 0;
  const clampedQuality = Math.max(0, Math.min(1, qualityScore));

  return Number(
    Math.max(
      0,
      clampedQuality * 0.5 + lexRatio * 0.35 + cleanRatio * 0.15 - noisePenalty - lowConfPenalty
    ).toFixed(3)
  );
}
