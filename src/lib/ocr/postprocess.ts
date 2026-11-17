import { getCompactLexicon } from './compactLexicon';

export interface HandwrittenLineInput {
  text: string;
  confidence: number;
  bbox?: { x0: number; y0: number; x1: number; y1: number };
}

export interface HandwrittenLineOutput extends HandwrittenLineInput {
  snappedWords?: Array<{ from: string; to: string }>;
}

function editDistanceAtMostOne(a: string, b: string): boolean {
  if (a === b) return true;
  const la = a.length;
  const lb = b.length;
  if (Math.abs(la - lb) > 1) return false;

  if (la === lb) {
    let diffs = 0;
    for (let i = 0; i < la; i++) {
      if (a[i] !== b[i]) {
        diffs++;
        if (diffs > 1) return false;
      }
    }
    return diffs === 1;
  }

  const shorter = la < lb ? a : b;
  const longer = la < lb ? b : a;
  let si = 0;
  let li = 0;
  let diffs = 0;

  while (si < shorter.length && li < longer.length) {
    if (shorter[si] === longer[li]) {
      si++;
      li++;
    } else {
      diffs++;
      if (diffs > 1) return false;
      li++;
    }
  }

  return true;
}

function snapWordToLexicon(word: string, lexicon: Set<string>): string {
  if (word.length < 3) return word;
  // Preserve capitalized proper nouns, acronyms, and tokens with digits
  if (/^[A-Z]/.test(word) || /\d/.test(word)) return word;

  const lower = word.toLowerCase();
  if (lexicon.has(lower)) return word;

  for (const candidate of lexicon) {
    if (Math.abs(candidate.length - lower.length) > 1) continue;
    if (candidate[0] !== lower[0]) continue;
    if (editDistanceAtMostOne(lower, candidate)) {
      return candidate;
    }
  }

  return word;
}

/**
 * Normalizes handwritten TrOCR output lines without applying printed-glyph homoglyph rules:
 * - collapses stray spaces and normalizes punctuation spacing
 * - repairs common handwritten time/date spacing (e.g. "3 pm" -> "3pm", "5 / 12" -> "5/12")
 * - snaps words to the compact lexicon ONLY when line confidence < 0.6 and edit distance <= 1
 */
export function postprocessHandwrittenLines(
  lines: HandwrittenLineInput[]
): HandwrittenLineOutput[] {
  const lexicon = getCompactLexicon();

  return lines
    .map((line) => {
      let text = (line.text || '')
        .replace(/\s+/g, ' ')
        .replace(/\s+([,.;:!?])/g, '$1')
        .replace(/(\d)\s+(am|pm)\b/gi, '$1$2')
        .replace(/(\d{1,2})\s*\/\s*(\d{1,2})/g, '$1/$2')
        .trim();

      const snappedWords: Array<{ from: string; to: string }> = [];

      if (line.confidence < 0.6 && text.length > 0) {
        const tokens = text.split(' ');
        const updatedTokens = tokens.map((tok) => {
          const match = tok.match(/^([^A-Za-z0-9]*)([A-Za-z0-9]+)([^A-Za-z0-9]*)$/);
          if (!match) return tok;
          const [, prefix, core, suffix] = match;
          const snapped = snapWordToLexicon(core, lexicon);
          if (snapped !== core) {
            snappedWords.push({ from: core, to: snapped });
          }
          return `${prefix}${snapped}${suffix}`;
        });
        text = updatedTokens.join(' ');
      }

      return {
        ...line,
        text,
        snappedWords: snappedWords.length > 0 ? snappedWords : undefined,
      };
    })
    .filter((line) => line.text.length > 0);
}
