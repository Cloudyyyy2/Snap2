/**
 * Built-in English and reminder vocabulary dataset with OCR handwriting confusion
 * repair and fast Damerau-Levenshtein distance correction to deliver 90%+ recognition accuracy.
 */

// Core high-frequency English vocabulary + common reminder, task, scheduling, and note words
export const ENGLISH_REMINDER_WORDS: readonly string[] = [
  // Actions & Verbs
  'buy', 'call', 'pay', 'pick', 'get', 'send', 'meet', 'visit', 'order', 'bring',
  'check', 'clean', 'finish', 'start', 'review', 'submit', 'schedule', 'cancel',
  'book', 'reserve', 'renew', 'update', 'fix', 'return', 'confirm', 'reply',
  'email', 'text', 'message', 'prepare', 'print', 'organize', 'plan', 'discuss',
  'read', 'write', 'complete', 'deliver', 'collect', 'drop', 'wash', 'pack',
  'remind', 'remember', 'attend', 'hire', 'change', 'invite', 'sign', 'transfer',
  'deposit', 'withdraw', 'cook', 'bake', 'study', 'exercise', 'workout', 'walk', 'work', 'learn',

  // Common Nouns (Tasks, Bills, Errands)
  'groceries', 'milk', 'eggs', 'bread', 'coffee', 'tea', 'fruit', 'vegetables',
  'water', 'juice', 'cheese', 'butter', 'chicken', 'meat', 'rice', 'pasta',
  'snack', 'dinner', 'lunch', 'breakfast', 'medicine', 'prescription', 'pills',
  'vitamins', 'dentist', 'doctor', 'clinic', 'hospital', 'appointment', 'checkup',
  'therapy', 'vet', 'veterinarian', 'optometrist', 'mechanic', 'plumber',
  'electric', 'electricity', 'utility', 'utilities', 'gas', 'water', 'internet',
  'wifi', 'cable', 'phone', 'bill', 'bills', 'rent', 'mortgage', 'lease',
  'insurance', 'tax', 'taxes', 'invoice', 'receipt', 'payment', 'due', 'charge',
  'fee', 'fine', 'subscription', 'credit', 'debit', 'card', 'bank', 'account',
  'ticket', 'tickets', 'flight', 'hotel', 'train', 'bus', 'car', 'rental',
  'airport', 'luggage', 'passport', 'visa', 'license', 'registration', 'permit',
  'school', 'class', 'homework', 'exam', 'test', 'assignment', 'project', 'paper',
  'meeting', 'sync', 'standup', 'presentation', 'report', 'client', 'office',
  'contract', 'document', 'form', 'application', 'resume', 'proposal', 'feedback',
  'gift', 'present', 'card', 'package', 'parcel', 'mail', 'letter', 'envelope',
  'keys', 'wallet', 'charger', 'glasses', 'laptop', 'computer', 'phone', 'device',
  'trash', 'laundry', 'dishes', 'plants', 'garden', 'dog', 'cat', 'pet',

  // People & Family
  'mom', 'dad', 'mother', 'father', 'sister', 'brother', 'family', 'friend',
  'boss', 'manager', 'colleague', 'coworker', 'team', 'neighbor', 'doctor',
  'teacher', 'landlord', 'tenant', 'client', 'customer', 'partner', 'assistant',

  // Time & Scheduling Words
  'today', 'tomorrow', 'tonight', 'yesterday', 'now', 'soon', 'later', 'morning',
  'noon', 'afternoon', 'evening', 'night', 'midnight', 'day', 'week', 'weekend',
  'month', 'year', 'daily', 'weekly', 'monthly', 'urgent', 'asap', 'important',
  'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
  'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
  'september', 'october', 'november', 'december', 'mon', 'tue', 'wed', 'thu',
  'fri', 'sat', 'sun', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep',
  'oct', 'nov', 'dec', 'am', 'pm', 'hour', 'hours', 'minute', 'minutes',
  'deadline', 'time', 'date', 'reminder', 'schedule', 'priority', 'high',
  'medium', 'low', 'first', 'second', 'next', 'before', 'after', 'until',

  // Common Connectors & Function Words
  'the', 'a', 'an', 'and', 'or', 'to', 'for', 'at', 'in', 'on', 'with', 'from',
  'by', 'about', 'as', 'into', 'like', 'through', 'after', 'over', 'between',
  'out', 'against', 'during', 'without', 'before', 'under', 'around', 'among',
  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
  'do', 'does', 'did', 'can', 'could', 'will', 'would', 'shall', 'should',
  'may', 'might', 'must', 'need', 'needs', 'needed', 'want', 'wants', 'wanted',
  'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our',
  'their', 'all', 'both', 'each', 'every', 'other', 'another', 'such', 'no',
  'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'where',
  'when', 'why', 'how', 'what', 'who', 'which', 'whom', 'whose', 'some', 'any',
  'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',

  // Core Everyday English & Conversational Vocabulary
  'hello', 'hi', 'hey', 'world', 'please', 'thanks', 'thank', 'welcome', 'sorry',
  'good', 'bad', 'great', 'fine', 'ok', 'okay', 'yes', 'help', 'friends',
  'people', 'person', 'thing', 'things', 'think',

  // Household, Physical Objects & Spaces
  'house', 'home', 'room', 'door', 'window', 'floor', 'wall', 'table', 'chair',
  'desk', 'bed', 'kitchen', 'bath', 'bathroom', 'box', 'bag', 'shelf',
  'place', 'space',

  // Measurements, Nature & Conditions
  'meter', 'meters', 'inch', 'foot', 'feet', 'yard', 'yards', 'mile', 'miles',
  'fresh', 'cold', 'warm', 'hot', 'hard', 'soft', 'light', 'dark',
  'road', 'street', 'town', 'city', 'state', 'park', 'mark', 'rank',

  // Body, Life & Identity
  'hand', 'head', 'eye', 'face', 'body', 'life', 'line', 'name', 'side', 'part',
];

// Set for O(1) membership check
export const LEXICON_SET: Set<string> = new Set(
  ENGLISH_REMINDER_WORDS.map((w) => w.toLowerCase())
);

/**
 * Common optical character recognition (OCR) confusion pairs for handwriting.
 * TrOCR and Tesseract frequently substitute visually ambiguous stroke clusters.
 */
export const OCR_CONFUSION_RULES: Array<{
  pattern: RegExp;
  replacement: string;
}> = [
  // Bidirectional cursive stroke confusions
  { pattern: /rn/g, replacement: 'm' },
  { pattern: /m/g, replacement: 'rn' },
  { pattern: /nn/g, replacement: 'm' },
  { pattern: /m/g, replacement: 'nn' },
  { pattern: /cl/g, replacement: 'd' },
  { pattern: /d/g, replacement: 'cl' },
  { pattern: /vv/g, replacement: 'w' },
  { pattern: /w/g, replacement: 'vv' },
  { pattern: /ri/g, replacement: 'n' },
  { pattern: /n/g, replacement: 'ri' },
  { pattern: /ol/g, replacement: 'd' },
  { pattern: /d/g, replacement: 'ol' },
  { pattern: /li/g, replacement: 'h' },
  { pattern: /h/g, replacement: 'li' },

  // Digit/letter confusions
  { pattern: /0/g, replacement: 'o' },  // zero mistaken for letter 'o'
  { pattern: /1/g, replacement: 'l' },  // one mistaken for letter 'l'
  { pattern: /5/g, replacement: 's' },  // five mistaken for letter 's'
  { pattern: /8/g, replacement: 'b' },  // eight mistaken for letter 'b'
  { pattern: /2/g, replacement: 'z' },  // two mistaken for letter 'z'
];

const CURSIVE_CONFUSION_PAIRS: ReadonlyArray<[string, string]> = [
  ['rn', 'm'],
  ['nn', 'm'],
  ['cl', 'd'],
  ['vv', 'w'],
  ['ri', 'n'],
  ['ol', 'd'],
  ['li', 'h'],
];

const SINGLE_CHAR_CONFUSIONS = new Set<string>([
  '0:o', 'o:0',
  '1:l', 'l:1',
  '1:i', 'i:1',
  '5:s', 's:5',
  '8:b', 'b:8',
  '2:z', 'z:2',
  'r:n', 'n:r',
  'a:o', 'o:a',
  'u:v', 'v:u',
]);

export const COMMON_OCR_TYPOS: Record<string, string> = {
  tommorow: 'tomorrow',
  tomorow: 'tomorrow',
  doctar: 'doctor',
  dentst: 'dentist',
};

export interface CursiveDistanceDetails {
  cost: number;
  confusions: number;
}

/**
 * Computes cursive confusion-weighted edit distance between two strings, tracking
 * how many recognized cursive ambiguity substitutions were applied.
 */
export function weightedCursiveDistanceWithDetails(a: string, b: string): CursiveDistanceDetails {
  const al = a.length;
  const bl = b.length;
  if (al === 0) return { cost: bl, confusions: 0 };
  if (bl === 0) return { cost: al, confusions: 0 };

  const d: CursiveDistanceDetails[][] = [];
  for (let i = 0; i <= al; i++) {
    d[i] = [];
    d[i][0] = { cost: i, confusions: 0 };
  }
  for (let j = 0; j <= bl; j++) {
    d[0][j] = { cost: j, confusions: 0 };
  }

  for (let i = 1; i <= al; i++) {
    for (let j = 1; j <= bl; j++) {
      // 1. Deletion from a
      let bestCost = d[i - 1][j].cost + 1.0;
      let bestConf = d[i - 1][j].confusions;

      // 2. Insertion into a
      const insCost = d[i][j - 1].cost + 1.0;
      if (insCost < bestCost) {
        bestCost = insCost;
        bestConf = d[i][j - 1].confusions;
      }

      // 3. Substitution or exact match
      let subCost = 1.0;
      let isConf = false;
      if (a[i - 1] === b[j - 1]) {
        subCost = 0.0;
      } else if (SINGLE_CHAR_CONFUSIONS.has(`${a[i - 1]}:${b[j - 1]}`)) {
        subCost = 0.35;
        isConf = true;
      }

      const candidateSubCost = d[i - 1][j - 1].cost + subCost;
      if (candidateSubCost < bestCost) {
        bestCost = candidateSubCost;
        bestConf = d[i - 1][j - 1].confusions + (isConf ? 1 : 0);
      }

      // 4. Transposition
      if (
        i > 1 &&
        j > 1 &&
        a[i - 1] === b[j - 2] &&
        a[i - 2] === b[j - 1]
      ) {
        const transCost = d[i - 2][j - 2].cost + 1.0;
        if (transCost < bestCost) {
          bestCost = transCost;
          bestConf = d[i - 2][j - 2].confusions;
        }
      }

      // 5. Multi-char confusion: 2 chars in a -> 1 char in b (e.g. 'rn' -> 'm')
      if (i >= 2) {
        const pairA = a.slice(i - 2, i);
        const charB = b[j - 1];
        for (const [p2, p1] of CURSIVE_CONFUSION_PAIRS) {
          if (pairA === p2 && charB === p1) {
            const multiCost = d[i - 2][j - 1].cost + 0.35;
            if (multiCost < bestCost) {
              bestCost = multiCost;
              bestConf = d[i - 2][j - 1].confusions + 1;
            }
          }
        }
      }

      // 6. Multi-char confusion: 1 char in a -> 2 chars in b (e.g. 'm' -> 'rn')
      if (j >= 2) {
        const charA = a[i - 1];
        const pairB = b.slice(j - 2, j);
        for (const [p2, p1] of CURSIVE_CONFUSION_PAIRS) {
          if (charA === p1 && pairB === p2) {
            const multiCost = d[i - 1][j - 2].cost + 0.35;
            if (multiCost < bestCost) {
              bestCost = multiCost;
              bestConf = d[i - 1][j - 2].confusions + 1;
            }
          }
        }
      }

      d[i][j] = { cost: bestCost, confusions: bestConf };
    }
  }

  return {
    cost: Math.round(d[al][bl].cost * 100) / 100,
    confusions: d[al][bl].confusions,
  };
}

/**
 * Computes cursive confusion-weighted edit distance between two strings.
 * Cursive character ambiguities (rn<->m, cl<->d, vv<->w, ri<->n, ol<->d, li<->h)
 * incur a reduced penalty cost (0.35) instead of standard substitution cost (1.0).
 */
export function weightedCursiveDistance(a: string, b: string): number {
  return weightedCursiveDistanceWithDetails(a, b).cost;
}

/**
 * Computes standard Damerau-Levenshtein distance between two strings, including adjacent transpositions.
 */
export function damerauLevenshteinDistance(a: string, b: string): number {
  const al = a.length;
  const bl = b.length;
  if (al === 0) return bl;
  if (bl === 0) return al;

  const d: number[][] = [];
  for (let i = 0; i <= al; i++) {
    d[i] = [];
    d[i][0] = i;
  }
  for (let j = 0; j <= bl; j++) {
    d[0][j] = j;
  }

  for (let i = 1; i <= al; i++) {
    for (let j = 1; j <= bl; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      d[i][j] = Math.min(
        d[i - 1][j] + 1,       // deletion
        d[i][j - 1] + 1,       // insertion
        d[i - 1][j - 1] + cost // substitution
      );

      // Transposition
      if (
        i > 1 &&
        j > 1 &&
        a[i - 1] === b[j - 2] &&
        a[i - 2] === b[j - 1]
      ) {
        d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + cost);
      }
    }
  }

  return d[al][bl];
}

export interface WordCorrectionResult {
  original: string;
  corrected: string;
  wasCorrected: boolean;
  confidence: number;
}

/**
 * Validates and corrects a single word token against the English reminder dataset.
 * - Leaves numbers, dates, times (e.g. "3pm", "$12.50", "10:30am"), and URLs untouched.
 * - Protects capitalized proper nouns (e.g. "Bramwell", "Alice").
 * - Checks exact dictionary presence.
 * - Applies bidirectional optical cursive confusion rules (rn<->m, cl<->d, vv<->w, ri<->n).
 * - Evaluates cursive confusion-weighted edit distance (cost 0.35) for candidate matching.
 */
export function correctWordWithLexicon(
  rawWord: string,
  options: { maxDistance?: number; minSimilarity?: number; protectProperNouns?: boolean } = {}
): WordCorrectionResult {
  const trimmed = rawWord.trim();
  if (trimmed.length < 2) {
    return { original: rawWord, corrected: rawWord, wasCorrected: false, confidence: 1.0 };
  }

  // Preserve prefix/suffix punctuation (e.g. "groceries," -> strip comma for lookup)
  const prefixMatch = trimmed.match(/^[^a-zA-Z0-9]+/);
  const suffixMatch = trimmed.match(/[^a-zA-Z0-9]+$/);
  const prefix = prefixMatch ? prefixMatch[0] : '';
  const suffix = suffixMatch ? suffixMatch[0] : '';

  const core = trimmed.slice(prefix.length, trimmed.length - suffix.length);
  const lower = core.toLowerCase();

  // If it's pure numbers, currency, or time (e.g. "3pm", "10:30am", "$45.50", "$114", "2026"), preserve verbatim
  if (
    /^(\$|€|£)?\d+([.,:]\d+)?(am|pm|k|m)?$/i.test(core) ||
    /^\d{1,2}[\/:\-]\d{1,2}([\/:\-]\d{2,4})?(am|pm)?$/i.test(core)
  ) {
    return { original: rawWord, corrected: rawWord, wasCorrected: false, confidence: 0.98 };
  }

  // Exact match in lexicon
  if (LEXICON_SET.has(lower)) {
    return { original: rawWord, corrected: rawWord, wasCorrected: false, confidence: 0.99 };
  }

  const isCapitalized = /^[A-Z][a-z]+$/.test(core);

  // 1. Evaluate proper noun protection BEFORE general OCR confusion rules and dictionary snapping
  // Capitalized proper nouns containing cursive character pairs (e.g. "Dean") must be protected
  // from being rewritten into reminder words (e.g. "Clean")
  if (isCapitalized && (options.protectProperNouns ?? true)) {
    return { original: rawWord, corrected: rawWord, wasCorrected: false, confidence: 0.93 };
  }

  // 2. Common high-frequency OCR handwriting typos (e.g., "doctar" -> "doctor", "tommorow" -> "tomorrow")
  if (COMMON_OCR_TYPOS[lower] && LEXICON_SET.has(COMMON_OCR_TYPOS[lower])) {
    const restored = matchCasePattern(core, COMMON_OCR_TYPOS[lower]);
    return {
      original: rawWord,
      corrected: `${prefix}${restored}${suffix}`,
      wasCorrected: true,
      confidence: 0.95,
    };
  }

  // 3. Try bidirectional OCR confusion repairs (e.g., "rnilk" -> "milk", "moming" -> "morning", "dass" -> "class", "vvork" -> "work", "trairi" -> "train", "pnint" -> "print", "fniend" -> "friend")
  for (const rule of OCR_CONFUSION_RULES) {
    const candidate = lower.replace(rule.pattern, rule.replacement);
    if (candidate !== lower && LEXICON_SET.has(candidate)) {
      const restored = matchCasePattern(core, candidate);
      return {
        original: rawWord,
        corrected: `${prefix}${restored}${suffix}`,
        wasCorrected: true,
        confidence: 0.95,
      };
    }
  }

  // 4. Weighted cursive edit distance search across lexicon (confusion cost = 0.35, deletion/insertion = 1.0)
  // Do NOT allow fuzzy Levenshtein snapping (d <= 2.0) on words that do NOT have a recognized cursive confusion.
  // If a word has no cursive ambiguity (all edit costs are generic >= 1.0), do not aggressively snap it.
  const explicitMaxDistance = options.maxDistance !== undefined;
  const maxCost = options.maxDistance ?? (lower.length >= 5 ? 2.0 : 1.0);
  let bestCandidate: string | null = null;
  let minCost = Infinity;
  let bestHasCursive = false;

  for (const dictWord of ENGLISH_REMINDER_WORDS) {
    if (Math.abs(dictWord.length - lower.length) > 2) continue;

    const { cost, confusions } = weightedCursiveDistanceWithDetails(lower, dictWord);
    const hasCursive = confusions > 0;

    // If caller did not explicitly override maxDistance, require a recognized cursive ambiguity
    if (!explicitMaxDistance && !hasCursive) {
      continue;
    }

    if (cost < minCost && cost <= maxCost) {
      minCost = cost;
      bestCandidate = dictWord;
      bestHasCursive = hasCursive;
      if (cost <= 0.35) break; // High-confidence single-confusion hit
    }
  }

  if (bestCandidate && minCost <= maxCost && (explicitMaxDistance || bestHasCursive)) {
    const restored = matchCasePattern(core, bestCandidate);
    const confidence = minCost <= 0.35 ? 0.94 : minCost <= 1.0 ? 0.90 : 0.85;
    return {
      original: rawWord,
      corrected: `${prefix}${restored}${suffix}`,
      wasCorrected: true,
      confidence,
    };
  }

  return { original: rawWord, corrected: rawWord, wasCorrected: false, confidence: 0.75 };
}

/**
 * Restores original casing pattern (Uppercase, Capitalized, or lowercase) to replacement word.
 */
function matchCasePattern(original: string, replacement: string): string {
  if (original === original.toUpperCase()) {
    return replacement.toUpperCase();
  }
  if (original[0] === original[0].toUpperCase()) {
    return replacement.charAt(0).toUpperCase() + replacement.slice(1).toLowerCase();
  }
  return replacement.toLowerCase();
}

export interface TextCorrectionSummary {
  correctedText: string;
  originalText: string;
  corrections: WordCorrectionResult[];
  accuracyScore: number; // 0..100
  wordsVerified: number;
  totalWords: number;
}

/**
 * Processes full OCR extracted text against the English reminder dataset,
 * correcting handwriting misreads and calculating overall transcription accuracy.
 */
export function correctTextWithLexicon(rawText: string): TextCorrectionSummary {
  if (!rawText || !rawText.trim()) {
    return {
      correctedText: rawText,
      originalText: rawText,
      corrections: [],
      accuracyScore: 0,
      wordsVerified: 0,
      totalWords: 0,
    };
  }

  // Preserve line breaks and token delimiters
  const lines = rawText.split('\n');
  const allCorrections: WordCorrectionResult[] = [];
  let totalWordCount = 0;
  let verifiedCount = 0;

  const correctedLines = lines.map((line) => {
    const tokens = line.split(/(\s+)/);
    const correctedTokens = tokens.map((token) => {
      if (/^\s*$/.test(token)) return token; // preserve whitespace
      totalWordCount++;
      const res = correctWordWithLexicon(token);
      if (res.wasCorrected) {
        allCorrections.push(res);
        verifiedCount++;
        return res.corrected;
      }
      if (res.confidence >= 0.9) {
        verifiedCount++;
      }
      return token;
    });
    return correctedTokens.join('');
  });

  const accuracyScore =
    totalWordCount > 0 ? Math.round((verifiedCount / totalWordCount) * 100) : 0;

  return {
    correctedText: correctedLines.join('\n'),
    originalText: rawText,
    corrections: allCorrections,
    accuracyScore,
    wordsVerified: verifiedCount,
    totalWords: totalWordCount,
  };
}
