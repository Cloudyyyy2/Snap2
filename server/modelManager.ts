import fs from 'fs';
import path from 'path';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import { isHandwritingModelAvailable } from './handwritingRecognizer';
import {
  getParseqDownloadUrls,
  getParseqModelDir,
  isParseqModelAvailable,
} from './parseqRecognizer';
import { isPrintedModelAvailable } from './printedRecognizer';

export type ModelDownloadState = 'idle' | 'downloading' | 'ready' | 'error';

export interface ModelProvisioningStatus {
  state: ModelDownloadState;
  handwritingModelState: ModelDownloadState;
  printedModelState: ModelDownloadState;
  downloadProgress: number;
  currentFile?: string;
  error?: string;
}

export interface EnsureOfflineModelsOptions {
  includeHandwriting?: boolean;
  includePrinted?: boolean;
  onProgress?: (status: ModelProvisioningStatus) => void;
}

const MODELS_ROOT = path.resolve(process.cwd(), 'public/models');
const TROCR_DIR = path.join(MODELS_ROOT, 'trocr-base-handwritten');
const TROCR_REPO = 'Xenova/trocr-base-handwritten';
const TROCR_FILES = [
  'config.json',
  'generation_config.json',
  'preprocessor_config.json',
  'special_tokens_map.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'vocab.json',
  'merges.txt',
  'onnx/encoder_model_quantized.onnx',
  'onnx/decoder_model_merged_quantized.onnx',
];

const TESSDATA_DIR = path.join(MODELS_ROOT, 'tessdata');
const TESSDATA_FULL_URL =
  'https://raw.githubusercontent.com/tesseract-ocr/tessdata/main/eng.traineddata';
const TESSDATA_FULL_MARKER = path.join(TESSDATA_DIR, '.tessdata-full');

let inFlightPromise: Promise<ModelProvisioningStatus> | null = null;
const status: ModelProvisioningStatus = {
  state: 'idle',
  handwritingModelState: isHandwritingModelAvailable() ? 'ready' : 'idle',
  printedModelState: isPrintedModelAvailable() ? 'ready' : 'idle',
  downloadProgress:
    isHandwritingModelAvailable() && isPrintedModelAvailable() ? 100 : 0,
};

function snapshotStatus(): ModelProvisioningStatus {
  const hwReady = isHandwritingModelAvailable() && isParseqModelAvailable();
  const prReady = isPrintedModelAvailable();
  return {
    ...status,
    handwritingModelState: isHandwritingModelAvailable() ? 'ready' : status.handwritingModelState,
    printedModelState: prReady ? 'ready' : status.printedModelState,
    state: hwReady && prReady ? 'ready' : status.state,
    downloadProgress: hwReady && prReady ? 100 : status.downloadProgress,
  };
}

export function getModelProvisioningStatus(): ModelProvisioningStatus {
  return snapshotStatus();
}

async function downloadFileAtomic(
  url: string,
  destPath: string,
  label: string,
  onFraction: (fraction: number) => void
): Promise<void> {
  if (fs.existsSync(destPath) && fs.statSync(destPath).size > 0) {
    onFraction(1);
    return;
  }

  fs.mkdirSync(path.dirname(destPath), { recursive: true });
  const tmpPath = `${destPath}.tmp-${process.pid}-${Date.now()}`;

  try {
    const response = await fetch(url, { redirect: 'follow' });
    if (!response.ok || !response.body) {
      throw new Error(
        `Failed to download ${label}: HTTP ${response.status} ${response.statusText}`
      );
    }

    const totalBytesHeader = response.headers.get('content-length');
    const totalBytes = totalBytesHeader ? parseInt(totalBytesHeader, 10) : 0;
    let downloadedBytes = 0;

    const webStream = response.body as ReadableStream<Uint8Array>;
    const reader = webStream.getReader();
    const nodeReadable = new Readable({
      async read() {
        try {
          const { done, value } = await reader.read();
          if (done) {
            this.push(null);
            return;
          }
          downloadedBytes += value.byteLength;
          if (totalBytes > 0) {
            onFraction(Math.min(0.99, downloadedBytes / totalBytes));
          }
          this.push(Buffer.from(value));
        } catch (err) {
          this.destroy(err instanceof Error ? err : new Error(String(err)));
        }
      },
    });

    const fileStream = fs.createWriteStream(tmpPath);
    await pipeline(nodeReadable, fileStream);
    fs.renameSync(tmpPath, destPath);
    onFraction(1);
  } catch (err) {
    try {
      if (fs.existsSync(tmpPath)) fs.unlinkSync(tmpPath);
    } catch {
      // ignore cleanup error
    }
    throw err;
  }
}

export async function ensureOfflineModels(
  options: EnsureOfflineModelsOptions = {}
): Promise<ModelProvisioningStatus> {
  const includeHandwriting = options.includeHandwriting !== false;
  const includePrinted = options.includePrinted !== false;

  const hwAlready =
    !includeHandwriting || (isHandwritingModelAvailable() && isParseqModelAvailable());
  const prAlready = !includePrinted || isPrintedModelAvailable();
  if (hwAlready && prAlready) {
    status.state = 'ready';
    status.handwritingModelState = isHandwritingModelAvailable() ? 'ready' : status.handwritingModelState;
    status.printedModelState = isPrintedModelAvailable() ? 'ready' : status.printedModelState;
    status.downloadProgress = 100;
    status.error = undefined;
    status.currentFile = undefined;
    const snap = snapshotStatus();
    options.onProgress?.(snap);
    return snap;
  }

  if (inFlightPromise) {
    return inFlightPromise;
  }

  inFlightPromise = (async () => {
    try {
      status.state = 'downloading';
      status.error = undefined;

      const tasks: Array<{
        group: 'handwriting' | 'printed';
        label: string;
        weight: number;
        run: (onFraction: (f: number) => void) => Promise<void>;
      }> = [];

      if (includeHandwriting && (!isHandwritingModelAvailable() || !isParseqModelAvailable())) {
        status.handwritingModelState = 'downloading';
        if (!isHandwritingModelAvailable()) {
          for (const rel of TROCR_FILES) {
            const dest = path.join(TROCR_DIR, rel);
            if (fs.existsSync(dest) && fs.statSync(dest).size > 0) continue;
            const weight = rel.endsWith('.onnx') ? 12 : 1;
            tasks.push({
              group: 'handwriting',
              label: `trocr-base-handwritten/${rel}`,
              weight,
              run: (onFrac) =>
                downloadFileAtomic(
                  `https://huggingface.co/${TROCR_REPO}/resolve/main/${rel}`,
                  dest,
                  `trocr-base-handwritten/${rel}`,
                  onFrac
                ),
            });
          }
        }
        if (!isParseqModelAvailable()) {
          const parseqDir = getParseqModelDir();
          for (const item of getParseqDownloadUrls()) {
            const dest = path.join(parseqDir, item.relPath);
            if (fs.existsSync(dest) && fs.statSync(dest).size > 0) continue;
            tasks.push({
              group: 'handwriting',
              label: `parseq/${item.relPath}`,
              weight: item.weight,
              run: (onFrac) =>
                downloadFileAtomic(item.url, dest, `parseq/${item.relPath}`, onFrac),
            });
          }
        }
      } else if (isHandwritingModelAvailable()) {
        status.handwritingModelState = 'ready';
      }

      if (includePrinted && !isPrintedModelAvailable()) {
        status.printedModelState = 'downloading';
        const rawDest = path.join(TESSDATA_DIR, 'eng.traineddata');
        const staleBest = path.join(TESSDATA_DIR, '.tessdata-best');
        tasks.push({
          group: 'printed',
          label: 'tessdata_full/eng.traineddata',
          weight: 5,
          run: async (onFrac) => {
            if (fs.existsSync(staleBest)) {
              try {
                fs.unlinkSync(staleBest);
              } catch {
                // ignore
              }
            }
            await downloadFileAtomic(
              TESSDATA_FULL_URL,
              rawDest,
              'tessdata_full/eng.traineddata',
              onFrac
            );
            fs.mkdirSync(TESSDATA_DIR, { recursive: true });
            fs.writeFileSync(
              TESSDATA_FULL_MARKER,
              `tessdata_full\n${new Date().toISOString()}\n`,
              'utf8'
            );
          },
        });
      } else if (isPrintedModelAvailable()) {
        status.printedModelState = 'ready';
      }

      const totalWeight = tasks.reduce((s, t) => s + t.weight, 0) || 1;
      let completedWeight = 0;

      for (const task of tasks) {
        status.currentFile = task.label;
        await task.run((fraction) => {
          const pct = Math.min(
            99,
            Math.round(((completedWeight + task.weight * fraction) / totalWeight) * 100)
          );
          status.downloadProgress = pct;
          options.onProgress?.(snapshotStatus());
        });
        completedWeight += task.weight;
        if (task.group === 'handwriting' && isHandwritingModelAvailable()) {
          status.handwritingModelState = 'ready';
        }
        if (task.group === 'printed' && isPrintedModelAvailable()) {
          status.printedModelState = 'ready';
        }
      }

      status.handwritingModelState = isHandwritingModelAvailable()
        ? 'ready'
        : includeHandwriting
          ? 'error'
          : status.handwritingModelState;
      status.printedModelState = isPrintedModelAvailable()
        ? 'ready'
        : includePrinted
          ? 'error'
          : status.printedModelState;
      status.currentFile = undefined;
      status.downloadProgress = 100;
      status.state = 'ready';

      const finalSnap = snapshotStatus();
      options.onProgress?.(finalSnap);
      return finalSnap;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      status.state = 'error';
      if (!isHandwritingModelAvailable() && includeHandwriting) {
        status.handwritingModelState = 'error';
      }
      if (!isPrintedModelAvailable() && includePrinted) {
        status.printedModelState = 'error';
      }
      status.error = msg;
      status.currentFile = undefined;
      // Reset inFlightPromise on error so retries work
      inFlightPromise = null;
      throw err;
    } finally {
      if (status.state === 'ready') {
        inFlightPromise = null;
      }
    }
  })();

  return inFlightPromise;
}
