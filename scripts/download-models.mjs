import fs from 'fs';
import path from 'path';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';

const MODEL_REPO = 'Xenova/trocr-small-handwritten';
const TARGET_DIR = path.resolve(process.cwd(), 'public/models/trocr-small-handwritten');

const FILES = [
  'config.json',
  'generation_config.json',
  'preprocessor_config.json',
  'special_tokens_map.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'sentencepiece.bpe.model',
  'onnx/encoder_model_quantized.onnx',
  'onnx/decoder_model_merged_quantized.onnx',
];

async function downloadFile(relPath) {
  const dest = path.join(TARGET_DIR, relPath);
  if (fs.existsSync(dest) && fs.statSync(dest).size > 0) {
    console.log(`[skip] ${relPath} already exists`);
    return;
  }

  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const url = `https://huggingface.co/${MODEL_REPO}/resolve/main/${relPath}`;
  console.log(`[download] ${url}`);

  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok || !response.body) {
    throw new Error(`Failed to download ${relPath}: HTTP ${response.status} ${response.statusText}`);
  }

  const tmpDest = `${dest}.tmp`;
  const fileStream = fs.createWriteStream(tmpDest);
  await pipeline(Readable.fromWeb(response.body), fileStream);
  fs.renameSync(tmpDest, dest);

  const sizeMb = (fs.statSync(dest).size / (1024 * 1024)).toFixed(2);
  console.log(`[done] ${relPath} (${sizeMb} MB)`);
}

async function main() {
  console.log(`Downloading ${MODEL_REPO} (quantized ONNX) to ${TARGET_DIR}...`);
  for (const file of FILES) {
    await downloadFile(file);
  }
  console.log('Handwriting model download complete.');
}

main().catch((err) => {
  console.error('Model download failed:', err);
  process.exit(1);
});
