import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

/**
 * Normalizes Unicode (NFC), line breaks (\r\n or \r -> \n), and Unicode horizontal whitespace
 * while preserving casing, punctuation, and numbers.
 */
export function normalizeWhitespace(text: string): string {
  return text
    .normalize('NFC')
    .replace(/\r\n?/g, '\n')
    .split('\n')
    .map((line) => line.replace(/[\u00A0\u2000-\u200B\u202F\u205F\u3000 \t]+/g, ' ').trim())
    .filter((line) => line.length > 0)
    .join('\n')
    .trim();
}

/**
 * Computes Levenshtein edit distance between two strings (Unicode code-point level).
 */
export function levenshteinDistance(a: string, b: string): number {
  const aChars = Array.from(a);
  const bChars = Array.from(b);
  const m = aChars.length;
  const n = bChars.length;
  if (m === 0) return n;
  if (n === 0) return m;

  let prev = new Int32Array(n + 1);
  let curr = new Int32Array(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;

  for (let i = 1; i <= m; i++) {
    curr[0] = i;
    const ca = aChars[i - 1];
    for (let j = 1; j <= n; j++) {
      const cost = ca === bChars[j - 1] ? 0 : 1;
      const del = prev[j] + 1;
      const ins = curr[j - 1] + 1;
      const sub = prev[j - 1] + cost;
      curr[j] = del < ins ? (del < sub ? del : sub) : ins < sub ? ins : sub;
    }
    const tmp = prev;
    prev = curr;
    curr = tmp;
  }
  return prev[n];
}

/**
 * Character Error Rate (CER) = editDistance(norm(hyp), norm(ref)) / max(1, norm(ref).length).
 * If caseInsensitive is true, compares lowercased strings (useful for cursive notebooks where
 * capitalization is ambiguous, whereas exact-case tests should pass caseInsensitive = false).
 */
export function computeCER(
  hypothesis: string,
  reference: string,
  options: { caseInsensitive?: boolean } = {}
): number {
  let h = normalizeWhitespace(hypothesis);
  let r = normalizeWhitespace(reference);
  if (options.caseInsensitive) {
    h = h.toLowerCase();
    r = r.toLowerCase();
  }
  const hChars = Array.from(h);
  const rChars = Array.from(r);
  if (rChars.length === 0) {
    return hChars.length;
  }
  return levenshteinDistance(h, r) / rChars.length;
}

/**
 * Character accuracy = max(0, 1 - CER).
 */
export function characterAccuracy(
  hypothesis: string,
  reference: string,
  options: { caseInsensitive?: boolean } = {}
): number {
  const cer = computeCER(hypothesis, reference, options);
  return Math.max(0, 1 - cer);
}

/**
 * Dynamically loads a module relative to repoRoot without triggering compile-time
 * `tsc --noEmit` errors if a planned M1-M3 module does not yet exist on disk.
 */
export async function loadOptionalModule(
  repoRoot: string,
  relPath: string
): Promise<Record<string, unknown> | null> {
  const absPath = path.resolve(repoRoot, relPath);
  if (!fs.existsSync(absPath)) {
    return null;
  }
  const url = pathToFileURL(absPath).href;
  const mod = await import(/* @vite-ignore */ url);
  return mod as Record<string, unknown>;
}

/**
 * Resolves the first existing function export from a module by candidate names.
 */
export function resolveExportFn<T extends (...args: any[]) => any>(
  mod: Record<string, unknown> | null,
  names: string[]
): T | null {
  if (!mod) return null;
  for (const name of names) {
    if (typeof mod[name] === 'function') {
      return mod[name] as T;
    }
  }
  return null;
}

export function assert(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(`Assertion failed: ${message}`);
  }
}

export function assertEqual<T>(actual: T, expected: T, message?: string): void {
  if (actual !== expected) {
    throw new Error(
      `${message ? `${message}: ` : ''}expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`
    );
  }
}

export function assertGreaterOrEqual(actual: number, min: number, message: string): void {
  if (!(actual >= min)) {
    throw new Error(`${message}: expected >= ${min}, got ${actual}`);
  }
}

export function assertLessOrEqual(actual: number, max: number, message: string): void {
  if (!(actual <= max)) {
    throw new Error(`${message}: expected <= ${max}, got ${actual}`);
  }
}

export function assertIncludes(haystack: string, needle: string, message?: string): void {
  if (!haystack.includes(needle)) {
    throw new Error(
      `${message ? `${message}: ` : ''}expected string to include ${JSON.stringify(needle)}; got ${JSON.stringify(haystack.slice(0, 300))}`
    );
  }
}

export function assertNotIncludes(haystack: string, needle: string, message?: string): void {
  if (haystack.includes(needle)) {
    throw new Error(
      `${message ? `${message}: ` : ''}expected string NOT to include ${JSON.stringify(needle)}; got ${JSON.stringify(haystack.slice(0, 300))}`
    );
  }
}

export function assertMatches(haystack: string, pattern: RegExp, message?: string): void {
  if (!pattern.test(haystack)) {
    throw new Error(
      `${message ? `${message}: ` : ''}expected string to match ${pattern}; got ${JSON.stringify(haystack.slice(0, 300))}`
    );
  }
}
