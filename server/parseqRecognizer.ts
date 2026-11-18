import fs from 'fs';
import path from 'path';
import sharp from 'sharp';

const PARSEQ_DIR = path.resolve(process.cwd(), 'public/models/parseq');
const PARSEQ_REPO = 'Felix92/onnxtr-parseq';
export const PARSEQ_FILES = ['config.json', 'model.onnx'] as const;

interface ParseqConfig {
  mean: [number, number, number];
  std: [number, number, number];
  input_shape: [number, number, number];
  vocab: string;
}

let sessionPromise: Promise<any> | null = null;
let cachedConfig: ParseqConfig | null = null;

export function getParseqModelDir(): string {
  return PARSEQ_DIR;
}

export function getParseqDownloadUrls(): Array<{ relPath: string; url: string; weight: number }> {
  return [
    {
      relPath: 'config.json',
      url: `https://huggingface.co/${PARSEQ_REPO}/raw/main/config.json`,
      weight: 1,
    },
    {
      relPath: 'model.onnx',
      url: `https://huggingface.co/${PARSEQ_REPO}/resolve/main/model.onnx`,
      weight: 10,
    },
  ];
}

export function isParseqModelAvailable(customDir?: string): boolean {
  const dir = customDir || PARSEQ_DIR;
  const cfgPath = path.join(dir, 'config.json');
  const modelPath = path.join(dir, 'model.onnx');
  return (
    fs.existsSync(cfgPath) &&
    fs.statSync(cfgPath).size > 0 &&
    fs.existsSync(modelPath) &&
    fs.statSync(modelPath).size > 1_000_000
  );
}

function loadParseqConfig(customDir?: string): ParseqConfig {
  if (cachedConfig && !customDir) return cachedConfig;
  const cfgPath = path.join(customDir || PARSEQ_DIR, 'config.json');
  const parsed = JSON.parse(fs.readFileSync(cfgPath, 'utf8')) as ParseqConfig;
  if (!customDir) cachedConfig = parsed;
  return parsed;
}

async function getParseqSession(customDir?: string) {
  const dir = customDir || PARSEQ_DIR;
  if (!isParseqModelAvailable(dir)) {
    throw new Error(`Local PARSeq model missing in ${dir}. Run "npm run models:handwriting" first.`);
  }
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await import('onnxruntime-node');
      return ort.InferenceSession.create(path.join(dir, 'model.onnx'));
    })().catch((err) => {
      sessionPromise = null;
      throw err;
    });
  }
  return sessionPromise;
}

export interface ParseqWordCrop {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

export interface ParseqWordResult {
  text: string;
  confidence: number;
}

/**
 * Runs PARSeq (Permuted Autoregressive Sequence Model) ONNX inference on pre-segmented
 * handwritten word crops. Preserves aspect ratio inside the model's [3, 32, 128] window
 * with clean white padding and cubic resampling.
 */
export async function recognizeWordsWithParseq(
  wordCrops: ParseqWordCrop[]
): Promise<ParseqWordResult[]> {
  if (wordCrops.length === 0) return [];

  const cfg = loadParseqConfig();
  const session = await getParseqSession();
  const ort = await import('onnxruntime-node');

  const [, targetH, targetW] = cfg.input_shape; // [3, 32, 128]
  const mean = cfg.mean;
  const std = cfg.std;
  const vocab = cfg.vocab;
  const eosIndex = vocab.length;
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const planeSize = targetH * targetW;

  const results: ParseqWordResult[] = [];

  for (const crop of wordCrops) {
    if (crop.width <= 0 || crop.height <= 0) {
      results.push({ text: '', confidence: 0 });
      continue;
    }

    const scale = Math.min(targetW / crop.width, targetH / crop.height);
    const nw = Math.max(1, Math.min(targetW, Math.round(crop.width * scale)));
    const nh = Math.max(1, Math.min(targetH, Math.round(crop.height * scale)));
    const left = Math.floor((targetW - nw) / 2);
    const top = Math.floor((targetH - nh) / 2);

    const { data: rgbBuf } = await sharp(Buffer.from(crop.data), {
      raw: { width: crop.width, height: crop.height, channels: 1 },
    })
      .resize(nw, nh, { kernel: sharp.kernel.cubic })
      .extend({
        top,
        bottom: targetH - nh - top,
        left,
        right: targetW - nw - left,
        background: { r: 255, g: 255, b: 255 },
      })
      .toColorspace('srgb')
      .raw()
      .toBuffer({ resolveWithObject: true });

    const channels = Math.max(1, Math.floor(rgbBuf.length / planeSize));
    const tensorData = new Float32Array(3 * planeSize);

    for (let y = 0; y < targetH; y++) {
      const rowOff = y * targetW;
      for (let x = 0; x < targetW; x++) {
        const pxIdx = (rowOff + x) * channels;
        const dstOff = rowOff + x;
        for (let c = 0; c < 3; c++) {
          const v = rgbBuf[pxIdx + (channels === 1 ? 0 : c)] / 255.0;
          tensorData[c * planeSize + dstOff] = (v - mean[c]) / std[c];
        }
      }
    }

    const inputTensor = new ort.Tensor('float32', tensorData, [1, 3, targetH, targetW]);
    const outputs = await session.run({ [inputName]: inputTensor });
    const logitsTensor = outputs[outputName];
    const dims = logitsTensor.dims as number[]; // [1, T, C]
    const logits = logitsTensor.data as Float32Array;
    const seqLen = dims[1];
    const numClasses = dims[2];

    let text = '';
    let probSum = 0;
    let charCount = 0;

    for (let t = 0; t < seqLen; t++) {
      const off = t * numClasses;
      let bestIdx = 0;
      let bestVal = -Infinity;
      for (let c = 0; c < numClasses; c++) {
        if (logits[off + c] > bestVal) {
          bestVal = logits[off + c];
          bestIdx = c;
        }
      }
      if (bestIdx === eosIndex) break;

      if (bestIdx < eosIndex) {
        let sumExp = 0;
        for (let c = 0; c < numClasses; c++) {
          sumExp += Math.exp(logits[off + c] - bestVal);
        }
        probSum += 1 / sumExp;
        charCount++;
        text += vocab[bestIdx];
      }
    }

    const confidence = charCount > 0 ? Number((probSum / charCount).toFixed(2)) : 0;
    results.push({ text: text.trim(), confidence });
  }

  return results;
}
