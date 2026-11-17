import { getCompactLexicon } from './compactLexicon';

export interface HandwrittenLineInput {
  text: string;
  confidence: number;
  bbox?: { x0: number; y0: number; x1: number; y1: number };
}

export interface HandwrittenLineOutput extends HandwrittenLineInput {
  snappedWords?: Array<{ from: string; to: string }>;
  hallucinated?: boolean;
}

/**
 * Detects repetitive decoder hallucination loops (e.g. "of the Government of the Government...",
 * "0 000 000 000...", "a b c d e f...") and strips or rejects degenerate output.
 */
export function detectAndCleanHallucination(raw: string): {
  text: string;
  isHallucination: boolean;
} {
  const cleaned = raw.trim().replace(/^[#*_~|]+\s*/, '').trim();
  if (!cleaned) return { text: '', isHallucination: true };

  // Purely numeric/zero loops like "0 000 000 000..." or "7 000 000 000"
  if (/^(?:0+\s+){4,}/.test(cleaned) || /(?:^|\s)0{2,}(?:\s+0{2,}){1,}/.test(cleaned)) {
    return { text: '', isHallucination: true };
  }

  // Standalone punctuation specks (e.g. '".', '-', '...')
  if (/^[^A-Za-z0-9]+$/.test(cleaned)) {
    return { text: '', isHallucination: true };
  }

  // Sequential single-letter alphabet gibberish like "a b c d e f"
  if (/^(?:[a-z]\s+){4,}[a-z]\b/i.test(cleaned)) {
    return { text: '', isHallucination: true };
  }

  const words = cleaned.split(/\s+/);
  if (words.length >= 6) {
    // Detect same word repeated consecutively 3+ times
    let maxRun = 1;
    let curRun = 1;
    for (let i = 1; i < words.length; i++) {
      if (words[i].toLowerCase() === words[i - 1].toLowerCase()) {
        curRun++;
        if (curRun > maxRun) maxRun = curRun;
      } else {
        curRun = 1;
      }
    }
    if (maxRun >= 3) {
      return { text: '', isHallucination: true };
    }

    // Detect repeated 2- or 3-word n-gram phrase loops ("of the Government of the Government")
    for (const n of [2, 3, 4]) {
      if (words.length < n * 3) continue;
      const counts = new Map<string, number>();
      for (let i = 0; i <= words.length - n; i++) {
        const gram = words
          .slice(i, i + n)
          .join(' ')
          .toLowerCase();
        const next = (counts.get(gram) || 0) + 1;
        counts.set(gram, next);
        if (next >= 3) {
          return { text: '', isHallucination: true };
        }
      }
    }

    // Check extreme token diversity collapse (e.g. > 65% duplicate words in long outputs)
    const uniqueRatio = new Set(words.map((w) => w.toLowerCase())).size / words.length;
    if (words.length >= 10 && uniqueRatio < 0.42) {
      return { text: '', isHallucination: true };
    }
  }

  return { text: cleaned, isHallucination: false };
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
 * - detects & drops repetitive decoder hallucinations
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
      const normalized = (line.text || '')
        .replace(/\s+/g, ' ')
        .replace(/\s+([,.;:!?])/g, '$1')
        .replace(/(\d)\s+(am|pm)\b/gi, '$1$2')
        .replace(/(\d{1,2})\s*\/\s*(\d{1,2})/g, '$1/$2')
        .trim();

      const { text: guardText, isHallucination } = detectAndCleanHallucination(normalized);
      if (isHallucination || !guardText) {
        return {
          ...line,
          text: '',
          confidence: 0.1,
          hallucinated: true,
        };
      }

      let text = guardText;
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
        hallucinated: false,
      };
    })
    .filter((line) => line.text.length > 0 && !line.hallucinated);
}
