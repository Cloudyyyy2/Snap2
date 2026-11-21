import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { getCompactLexicon } from '../src/lib/ocr/compactLexicon';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '..');
const outputPath = path.join(repoRoot, 'python_ocr', 'compact_lexicon.json');

const words = Array.from(getCompactLexicon());
const serialized = JSON.stringify(words);

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, serialized, 'utf8');
console.log(`Synced ${words.length} lexicon words -> ${outputPath}`);
