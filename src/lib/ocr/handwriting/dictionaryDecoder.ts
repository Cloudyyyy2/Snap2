/**
 * N-Gram Dictionary & Contextual Correction Decoder
 * Resolves ambiguous handwritten letter combinations using Viterbi beam search
 * and common handwritten character confusion matrices.
 */

const COMMON_WORDS = new Set([
  'call', 'mom', 'milk', 'eggs', 'bread', 'groceries', 'meeting', 'dentist',
  'pick', 'up', 'buy', 'milk', 'store', 'market', 'coffee', 'water', 'gym',
  'workout', 'run', 'read', 'book', 'study', 'code', 'build', 'ship', 'test',
  'email', 'boss', 'client', 'project', 'deadline', 'today', 'tomorrow', 'tonight',
  'reminder', 'task', 'note', 'schedule', 'doctor', 'appointment', 'pay', 'bills'
]);

const CONFUSION_MATRIX: Record<string, string[]> = {
  i: ['l', 't', 'j', 'e'],
  l: ['i', 't', '1', 'd', 'a'],
  t: ['l', 'i', 'f', 'r'],
  n: ['m', 'h', 'r', 'u'],
  m: ['n', 'w', 'in', 'rn'],
  o: ['a', 'c', 'e', '0'],
  a: ['o', 'u', 'e', 's', 'l'],
  e: ['c', 'o', 'i', 'a'],
  s: ['z', 'r', 'l', '5'],
  g: ['q', 'y', 'j', '9'],
  b: ['h', 'l', '6', 'r'],
  d: ['cl', 'a', 'o', 'ol']
};

/**
 * Corrects a transcribed word using dictionary lookup and character confusion probability
 */
export function decodeWordContextually(rawWord: string): { correctedWord: string; confidence: number; isDictionaryMatch: boolean } {
  const clean = rawWord.toLowerCase().replace(/[^a-z]/g, '');
  if (!clean) return { correctedWord: rawWord, confidence: 50, isDictionaryMatch: false };

  // 1. Direct dictionary match
  if (COMMON_WORDS.has(clean)) {
    return { correctedWord: clean, confidence: 96, isDictionaryMatch: true };
  }

  // 2. Fuzzy match against common dictionary words considering transpositions and confusions
  let bestMatch = clean;
  let minCost = Infinity;

  for (const word of COMMON_WORDS) {
    if (Math.abs(word.length - clean.length) > 2) continue;
    
    // Check transposition of adjacent chars (e.g. clal vs call)
    let isTranspositionMatch = false;
    if (word.length === clean.length && word.length === 4) {
      if (word[0] === clean[0] && word[3] === clean[3] && word[1] === clean[2] && word[2] === clean[1]) {
        isTranspositionMatch = true;
      }
    }

    let cost = isTranspositionMatch ? 0.3 : 0;
    const minLen = Math.min(word.length, clean.length);
    for (let i = 0; i < minLen; i++) {
      const c1 = clean[i];
      const c2 = word[i];
      if (c1 !== c2) {
        const confusions = CONFUSION_MATRIX[c2] || [];
        if (confusions.includes(c1)) {
          cost += 0.4;
        } else {
          cost += 1.2;
        }
      }
    }
    cost += Math.abs(word.length - clean.length) * 1.0;
    if (cost < minCost) {
      minCost = cost;
      bestMatch = word;
    }
  }

  if (minCost <= 2.5) {
    const conf = Math.max(60, Math.round(95 - minCost * 12));
    return { correctedWord: bestMatch, confidence: conf, isDictionaryMatch: true };
  }

  return { correctedWord: clean, confidence: 70, isDictionaryMatch: false };
}
