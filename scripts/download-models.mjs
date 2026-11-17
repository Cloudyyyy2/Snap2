import fs from 'fs';
import path from 'path';
import zlib from 'zlib';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';

const useSmallFlag = process.argv.includes('--small');
const MODEL_REPO = useSmallFlag
  ? 'Xenova/trocr-small-handwritten'
  : 'Xenova/trocr-base-handwritten';
const TARGET_DIR_NAME = useSmallFlag
  ? 'trocr-small-handwritten'
  : 'trocr-base-handwritten';
const TARGET_DIR = path.resolve(process.cwd(), 'public/models', TARGET_DIR_NAME);

const COMMON_FILES = [
  'config.json',
  'generation_config.json',
  'preprocessor_config.json',
  'special_tokens_map.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'onnx/encoder_model_quantized.onnx',
  'onnx/decoder_model_merged_quantized.onnx',
];

const MODEL_TOKENIZER_FILES = useSmallFlag
  ? ['sentencepiece.bpe.model']
  : ['vocab.json', 'merges.txt'];

const FILES = [...COMMON_FILES, ...MODEL_TOKENIZER_FILES];

const TESSDATA_DIR = path.resolve(process.cwd(), 'public/models/tessdata');
const TESSDATA_FULL_URL =
  'https://raw.githubusercontent.com/tesseract-ocr/tessdata/main/eng.traineddata';
const TESSDATA_FULL_MARKER = path.join(TESSDATA_DIR, '.tessdata-full');

async function downloadFile(relPath) {
  const dest = path.join(TARGET_DIR, relPath);
  if (fs.existsSync(dest) && fs.statSync(dest).size > 0) {
    console.log(`[skip] ${TARGET_DIR_NAME}/${relPath} already exists`);
    return;
  }

  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const url = `https://huggingface.co/${MODEL_REPO}/resolve/main/${relPath}`;
  console.log(`[download] ${url}`);

  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok || !response.body) {
    throw new Error(
      `Failed to download ${relPath}: HTTP ${response.status} ${response.statusText}`
    );
  }

  const tmpDest = `${dest}.tmp`;
  const fileStream = fs.createWriteStream(tmpDest);
  await pipeline(Readable.fromWeb(response.body), fileStream);
  fs.renameSync(tmpDest, dest);

  const sizeMb = (fs.statSync(dest).size / (1024 * 1024)).toFixed(2);
  console.log(`[done] ${TARGET_DIR_NAME}/${relPath} (${sizeMb} MB)`);
}

async function downloadTessdataFull() {
  const rawDest = path.join(TESSDATA_DIR, 'eng.traineddata');
  const gzDest = path.join(TESSDATA_DIR, 'eng.traineddata.gz');
  const staleBestMarker = path.join(TESSDATA_DIR, '.tessdata-best');

  if (
    fs.existsSync(TESSDATA_FULL_MARKER) &&
    fs.existsSync(rawDest) &&
    fs.statSync(rawDest).size > 20_000_000
  ) {
    console.log('[skip] tessdata-full/eng.traineddata already installed');
    return;
  }

  fs.mkdirSync(TESSDATA_DIR, { recursive: true });
  if (fs.existsSync(staleBestMarker)) fs.unlinkSync(staleBestMarker);
  console.log(`[download] ${TESSDATA_FULL_URL}`);

  const response = await fetch(TESSDATA_FULL_URL, { redirect: 'follow' });
  if (!response.ok || !response.body) {
    throw new Error(
      `Failed to download tessdata/eng.traineddata: HTTP ${response.status} ${response.statusText}`
    );
  }

  const tmpRaw = `${rawDest}.tmp`;
  const fileStream = fs.createWriteStream(tmpRaw);
  await pipeline(Readable.fromWeb(response.body), fileStream);
  fs.renameSync(tmpRaw, rawDest);

  const rawBuf = fs.readFileSync(rawDest);
  const gzBuf = zlib.gzipSync(rawBuf, { level: 6 });
  fs.writeFileSync(gzDest, gzBuf);
  fs.writeFileSync(TESSDATA_FULL_MARKER, `tessdata_full\n${new Date().toISOString()}\n`, 'utf8');

  const rawMb = (rawBuf.length / (1024 * 1024)).toFixed(2);
  const gzMb = (gzBuf.length / (1024 * 1024)).toFixed(2);
  console.log(`[done] tessdata/eng.traineddata (${rawMb} MB raw / ${gzMb} MB gz)`);
}

async function main() {
  console.log(`Downloading ${MODEL_REPO} (quantized ONNX) to ${TARGET_DIR}...`);
  for (const file of FILES) {
    await downloadFile(file);
  }

  await downloadTessdataFull();

  console.log(
    `Offline OCR models (${TARGET_DIR_NAME} + tessdata_full) ready.`
  );
}

main().catch((err) => {
  console.error('Model download failed:', err);
  process.exit(1);
});
