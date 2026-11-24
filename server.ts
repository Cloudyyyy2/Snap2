import express from 'express';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import { createRequire } from 'module';
import type webpushType from 'web-push';
import dotenv from 'dotenv';
import { spawn, ChildProcess } from 'child_process';
import crypto from 'crypto';
import fs from 'fs';
import http from 'http';
import https from 'https';
import os from 'os';
import path from 'path';
import {
  isHandwritingModelAvailable,
  getActiveHandwritingModelName,
  OcrProfile,
} from './server/handwritingRecognizer';
import { isPrintedModelAvailable, getActivePrintedModelName } from './server/printedRecognizer';
import { parseReminderLocally, LocalCaptureMode } from './server/localReminderParser';
import { ensureOfflineModels, getModelProvisioningStatus } from './server/modelManager';
import { correctTextWithLexicon, correctGroceryContext } from './src/lib/ocr/wordDataset';

const require = createRequire(import.meta.url);
const webpush = require('web-push') as typeof webpushType;

dotenv.config();
dotenv.config({ path: '.env.production' });

const app = express();
app.use(express.json({ limit: '15mb' }));
app.use((_req, res, next) => {
  res.setHeader('Permissions-Policy', 'camera=*');
  next();
});

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

const vapidPublicKey = process.env.VITE_VAPID_PUBLIC_KEY || '';
const vapidPrivateKey = process.env.VAPID_PRIVATE_KEY || '';
let pushConfigured = Boolean(vapidPublicKey && vapidPrivateKey);

if (pushConfigured) {
  try {
    webpush.setVapidDetails(
      'mailto:notifications@snapremind.app',
      vapidPublicKey,
      vapidPrivateKey
    );
  } catch (err) {
    console.warn('Invalid VAPID keys, disabling Web Push:', err);
    pushConfigured = false;
  }
}

const isOfflineOnly = () => process.env.OFFLINE_ONLY === '1';

// Python Local OCR FastAPI sidecar management (strict Python-only local OCR)
const PYTHON_OCR_PORT = process.env.PYTHON_OCR_PORT || '8001';
const PYTHON_OCR_URL = process.env.PYTHON_OCR_URL || `http://127.0.0.1:${PYTHON_OCR_PORT}`;
let pythonChild: ChildProcess | null = null;
let pythonStartPromise: Promise<boolean> | null = null;

function resolvePythonExecutable(): string {
  const winVenv = path.resolve(process.cwd(), '.venv', 'Scripts', 'python.exe');
  if (fs.existsSync(winVenv)) return winVenv;
  const unixVenv = path.resolve(process.cwd(), '.venv', 'bin', 'python');
  if (fs.existsSync(unixVenv)) return unixVenv;
  return process.platform === 'win32' ? 'python' : 'python3';
}

async function isPythonOcrHealthy(): Promise<boolean> {
  try {
    const resp = await fetch(`${PYTHON_OCR_URL}/health`, { signal: AbortSignal.timeout(1500) });
    if (!resp.ok) return false;
    const data = await resp.json();
    return data?.status === 'ok';
  } catch {
    return false;
  }
}

async function ensurePythonOcrSidecar(): Promise<boolean> {
  if (await isPythonOcrHealthy()) return true;
  if (pythonStartPromise) return pythonStartPromise;

  pythonStartPromise = (async () => {
    const pyExe = resolvePythonExecutable();
    pythonChild = spawn(
      pyExe,
      ['-m', 'uvicorn', 'python_ocr.server:app', '--host', '127.0.0.1', '--port', PYTHON_OCR_PORT],
      {
        cwd: process.cwd(),
        env: {
          ...process.env,
          HF_HUB_OFFLINE: '1',
          TRANSFORMERS_OFFLINE: '1',
          PYTHONUNBUFFERED: '1',
        },
        stdio: ['ignore', 'pipe', 'pipe'],
      }
    );

    pythonChild.stdout?.on('data', (d) => {
      const msg = String(d).trim();
      if (msg) console.log(`[python_ocr] ${msg}`);
    });
    pythonChild.stderr?.on('data', (d) => {
      const msg = String(d).trim();
      if (msg) console.warn(`[python_ocr] ${msg}`);
    });
    pythonChild.on('exit', () => {
      pythonChild = null;
      pythonStartPromise = null;
    });

    for (let i = 0; i < 30; i++) {
      await new Promise((r) => setTimeout(r, 500));
      if (await isPythonOcrHealthy()) return true;
    }
    return false;
  })();

  const ok = await pythonStartPromise;
  if (!ok) pythonStartPromise = null;
  return ok;
}

const cleanupPythonSidecar = () => {
  if (pythonChild && !pythonChild.killed) {
    pythonChild.kill();
    pythonChild = null;
  }
};
process.on('exit', cleanupPythonSidecar);
process.on('SIGINT', () => {
  cleanupPythonSidecar();
  process.exit(0);
});
process.on('SIGTERM', () => {
  cleanupPythonSidecar();
  process.exit(0);
});

async function callPythonOcr(params: {
  image: string;
  ocrProfile: OcrProfile;
  exportStages: boolean;
  captureMode: LocalCaptureMode;
}) {
  const ready = await ensurePythonOcrSidecar();
  if (!ready) {
    const err: any = new Error(
      'Local Python OCR sidecar is unavailable. Ensure .venv is set up and python_ocr/server.py can start.'
    );
    err.status = 503;
    throw err;
  }

  const resp = await fetch(`${PYTHON_OCR_URL}/ocr`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image: params.image,
      ocrProfile: params.ocrProfile,
      captureMode: params.captureMode,
      exportStages: params.exportStages,
    }),
  });

  const data = await resp.json();
  if (!resp.ok || !data?.success) {
    const err: any = new Error(data?.detail || data?.error || `Python OCR failed (HTTP ${resp.status})`);
    err.status = resp.status >= 400 && resp.status < 600 ? resp.status : 500;
    throw err;
  }

  return data;
}

// Disk-persisted store for active browser push subscriptions
const PUSH_STORE_PATH = path.resolve(process.cwd(), 'data', 'push-subscriptions.json');
const pushSubscriptions = new Map<string, webpushType.PushSubscription>();

function loadPushSubscriptions() {
  try {
    if (fs.existsSync(PUSH_STORE_PATH)) {
      const raw = JSON.parse(fs.readFileSync(PUSH_STORE_PATH, 'utf8'));
      if (Array.isArray(raw)) {
        for (const sub of raw) {
          if (sub?.endpoint) pushSubscriptions.set(sub.endpoint, sub);
        }
      }
    }
  } catch (err) {
    console.warn('Failed to load persisted push subscriptions:', err);
  }
}

function savePushSubscriptions() {
  try {
    fs.mkdirSync(path.dirname(PUSH_STORE_PATH), { recursive: true });
    fs.writeFileSync(
      PUSH_STORE_PATH,
      JSON.stringify(Array.from(pushSubscriptions.values()), null, 2),
      'utf8'
    );
  } catch (err) {
    console.warn('Failed to persist push subscriptions:', err);
  }
}

loadPushSubscriptions();

app.get('/api/health', (_req, res) => {
  const offlineOnly = isOfflineOnly();
  res.json({
    status: 'ok',
    offlineOnly,
    aiConfigured: !offlineOnly && Boolean(process.env.GROQ_API_KEY || process.env.GEMINI_API_KEY),
    groqConfigured: !offlineOnly && Boolean(process.env.GROQ_API_KEY),
    geminiConfigured: !offlineOnly && Boolean(process.env.GEMINI_API_KEY),
    supabaseConfigured: Boolean(process.env.VITE_SUPABASE_URL && process.env.VITE_SUPABASE_ANON_KEY),
    pushConfigured,
  });
});

// Safe public client configuration (never exposes private keys or DB URLs)
app.get('/api/config', (_req, res) => {
  const offlineOnly = isOfflineOnly();
  res.json({
    offlineOnly,
    supabaseUrl: process.env.VITE_SUPABASE_URL || '',
    supabaseAnonKey: process.env.VITE_SUPABASE_ANON_KEY || '',
    vapidPublicKey,
    groqConfigured: !offlineOnly && Boolean(process.env.GROQ_API_KEY),
    pushConfigured,
  });
});

app.get('/api/local-ocr/status', async (_req, res) => {
  const offlineOnly = isOfflineOnly();
  const hwName = getActiveHandwritingModelName();
  const printedName = getActivePrintedModelName();
  const hwReady = isHandwritingModelAvailable();
  const prReady = isPrintedModelAvailable();
  const prov = getModelProvisioningStatus();
  const pythonOcrReady = await isPythonOcrHealthy();

  const handwritingModel = hwReady
    ? 'ready'
    : prov.handwritingModelState === 'downloading'
      ? 'downloading'
      : 'missing';
  const printedModel = prReady
    ? 'ready'
    : prov.printedModelState === 'downloading'
      ? 'downloading'
      : 'missing';

  res.json({
    offlineOnly,
    pythonOcrReady,
    engine: 'python-local-ocr',
    handwritingModel,
    printedModel,
    handwritingModelState: hwReady ? 'ready' : prov.handwritingModelState,
    printedModelState: prReady ? 'ready' : prov.printedModelState,
    downloadProgress: prov.downloadProgress,
    currentFile: prov.currentFile,
    downloadError: prov.error,
    handwritingModelId: hwName,
    printedModelId: printedName,
    modelId: `${hwName}+${printedName}`,
    groqConfigured: !offlineOnly && Boolean(process.env.GROQ_API_KEY),
    geminiConfigured: !offlineOnly && Boolean(process.env.GEMINI_API_KEY),
    visionConfigured: !offlineOnly && Boolean(process.env.GROQ_API_KEY || process.env.GEMINI_API_KEY),
  });
});

app.post('/api/local-ocr', async (req, res) => {
  try {
    const { image, ocrProfile, exportStages, prompt, captureMode } = req.body || {};
    if (!image || typeof image !== 'string') {
      return res.status(400).json({
        success: false,
        error: 'Please provide a base64 image for local OCR.',
      });
    }

    const validProfile: OcrProfile =
      ocrProfile === 'printed' || ocrProfile === 'handwriting' || ocrProfile === 'auto'
        ? ocrProfile
        : 'auto';
    const validMode: LocalCaptureMode =
      captureMode === 'grocery' ||
      captureMode === 'receipt' ||
      captureMode === 'recipe' ||
      captureMode === 'fridge'
        ? captureMode
        : 'auto';

    if (!isOfflineOnly() && process.env.SKIP_AUTO_MODEL_DOWNLOAD !== '1') {
      if (validProfile !== 'printed' && !isHandwritingModelAvailable()) {
        await ensureOfflineModels({
          includeHandwriting: true,
          includePrinted: validProfile === 'auto',
        });
      } else if (validProfile === 'printed' && !isPrintedModelAvailable()) {
        await ensureOfflineModels({
          includeHandwriting: false,
          includePrinted: true,
        });
      }
    } else {
      const hasNeeded =
        validProfile === 'printed'
          ? isPrintedModelAvailable()
          : validProfile === 'handwriting'
            ? isHandwritingModelAvailable()
            : isHandwritingModelAvailable() || isPrintedModelAvailable();
      if (!hasNeeded) {
        return res.status(503).json({
          success: false,
          error:
            'Offline local OCR models are missing under public/models/. Run "npm run models:handwriting" while online once.',
        });
      }
    }

    const result = await callPythonOcr({
      image,
      ocrProfile: validProfile,
      exportStages: exportStages !== false,
      captureMode: validMode,
    });

    const deterministicData = result.text?.trim()
      ? parseReminderLocally(
          result.text,
          typeof prompt === 'string' ? prompt : undefined,
          result.detectedStyle === 'handwriting' || result.detectedStyle === 'mixed'
            ? ['handwritten']
            : ['ocr'],
          validMode
        )
      : undefined;

    res.json({
      success: true,
      offlineOnly: isOfflineOnly(),
      exportDirName: result.exportDirName || (result.exportDir ? path.basename(result.exportDir) : undefined),
      ...result,
      data: deterministicData,
    });
  } catch (error: any) {
    console.error('Error running local Python OCR:', error);
    const status = typeof error?.status === 'number' ? error.status : 500;
    res.status(status).json({
      success: false,
      error: error?.message || 'Failed to process image with local Python OCR.',
    });
  }
});

// Automatic offline training dataset collector (saves user-corrected OCR line crops locally)
app.post('/api/training/save-sample', async (req, res) => {
  try {
    const { exportDirName, lineIndex, originalText, correctedText } = req.body || {};
    const cleanLabel = typeof correctedText === 'string' ? correctedText.replace(/[\r\n\t]+/g, ' ').trim() : '';
    if (!cleanLabel) {
      return res.status(400).json({ success: false, error: 'correctedText is required' });
    }
    const safeDir = typeof exportDirName === 'string' ? path.basename(exportDirName.trim()) : '';
    const idxNum = typeof lineIndex === 'number' && Number.isInteger(lineIndex) && lineIndex >= 0 ? lineIndex : -1;
    if (!safeDir || idxNum < 0) {
      return res.json({ success: true, saved: false, reason: 'missing-line-crop-reference' });
    }

    const lineNumStr = String(idxNum + 1).padStart(2, '0');
    const srcCropPath = path.join(process.cwd(), 'ocr-exports', safeDir, `06_line_${lineNumStr}.png`);
    if (!fs.existsSync(srcCropPath)) {
      return res.json({ success: true, saved: false, reason: 'crop-not-found' });
    }

    const cropBytes = fs.readFileSync(srcCropPath);
    const hash = crypto
      .createHash('sha256')
      .update(cropBytes)
      .update('|')
      .update(cleanLabel)
      .digest('hex')
      .slice(0, 12);

    const realDir = path.join(process.cwd(), 'training', 'dataset', 'real');
    const imagesDir = path.join(realDir, 'images');
    fs.mkdirSync(imagesDir, { recursive: true });

    const fname = `real_${hash}.png`;
    const dstPath = path.join(imagesDir, fname);
    const manifestPath = path.join(realDir, 'labels.tsv');

    if (fs.existsSync(dstPath)) {
      return res.json({ success: true, saved: false, duplicate: true, file: `images/${fname}` });
    }

    fs.writeFileSync(dstPath, cropBytes);
    if (!fs.existsSync(manifestPath)) {
      fs.writeFileSync(manifestPath, 'rel_path\tlabel\tsource\twriter_id\toriginal_text\n', 'utf-8');
    }
    const cleanOrig = typeof originalText === 'string' ? originalText.replace(/[\r\n\t]+/g, ' ').trim() : '';
    const row = `images/${fname}\t${cleanLabel}\tsnap2_real\t${safeDir}\t${cleanOrig}\n`;
    fs.appendFileSync(manifestPath, row, 'utf-8');

    res.json({
      success: true,
      saved: true,
      file: `images/${fname}`,
      originalText: typeof originalText === 'string' ? originalText : undefined,
    });
  } catch (error: any) {
    console.error('Error saving training sample:', error);
    res.status(500).json({ success: false, error: error?.message || 'Failed to save training sample' });
  }
});

async function extractWithGroq(image: string | null, instructionText: string) {
  const groqKey = process.env.GROQ_API_KEY;
  if (!groqKey) return null;

  const model = process.env.GROQ_VISION_MODEL || 'qwen/qwen3.8-27b';

  const contentParts: Array<Record<string, unknown>> = [];
  if (image && typeof image === 'string') {
    const dataUrl = image.startsWith('data:') ? image : `data:image/jpeg;base64,${image}`;
    contentParts.push({
      type: 'image_url',
      image_url: { url: dataUrl },
    });
  }
  contentParts.push({
    type: 'text',
    text: `${instructionText}
First, carefully read every word in the image (handwritten cursive, block print, printed text, screenshots, bills, or receipts) and transcribe it verbatim into "rawText" preserving line breaks. Do NOT invent, guess unrelated sentences, or hallucinate text not clearly visible.
Return ONLY valid JSON matching this schema:
{
  "rawText": "Exact verbatim line-by-line text visible in the image",
  "title": "Clear concise reminder title",
  "description": "Summary of important details, amounts, or instructions",
  "dueDate": "ISO-8601 date-time string",
  "category": "Bills | Events | Tasks | Travel | Work | Personal | Health",
  "priority": "high | medium | low",
  "checklist": [
    { "id": "1", "text": "Action step", "completed": false }
  ],
  "tags": ["tag1", "tag2"]
}`,
  });

  const resp = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${groqKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      messages: [
        {
          role: 'system',
          content: 'You are SnapRemind, a high-accuracy OCR and reminder extraction assistant. Transcribe all visible handwritten or printed words faithfully into rawText before structuring reminder JSON. Always respond with valid JSON only.',
        },
        {
          role: 'user',
          content: contentParts,
        },
      ],
      response_format: { type: 'json_object' },
      temperature: 0.1,
    }),
    signal: AbortSignal.timeout(25000),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    throw new Error(`Groq API error (${resp.status}): ${errText}`);
  }

  const json = (await resp.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  let rawContent = json.choices?.[0]?.message?.content || '';
  rawContent = rawContent.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();

  const codeBlockMatch = rawContent.match(/```json\s*([\s\S]*?)\s*```/i);
  if (codeBlockMatch) {
    rawContent = codeBlockMatch[1];
  }

  return JSON.parse(rawContent);
}

async function generateWithGeminiRetry(parts: Array<Record<string, unknown>>) {
  const models = ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-2.5-pro'];
  let lastError: unknown;

  for (const model of models) {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const response = await ai.models.generateContent({
          model,
          contents: { parts },
          config: {
            abortSignal: AbortSignal.timeout(25000),
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                rawText: {
                  type: Type.STRING,
                  description: 'Exact verbatim line-by-line text visible in the image (printed or handwritten)',
                },
                title: { type: Type.STRING, description: 'Clear concise reminder title' },
                description: {
                  type: Type.STRING,
                  description: 'Summary of important details, amounts, or instructions',
                },
                dueDate: { type: Type.STRING, description: 'ISO-8601 date-time string' },
                category: {
                  type: Type.STRING,
                  description: 'One of: Bills, Events, Tasks, Travel, Work, Personal, Health',
                },
                priority: {
                  type: Type.STRING,
                  description: 'One of: high, medium, low',
                },
                checklist: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      id: { type: Type.STRING },
                      text: { type: Type.STRING },
                      completed: { type: Type.BOOLEAN },
                    },
                    required: ['id', 'text', 'completed'],
                  },
                },
                tags: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                },
              },
              required: ['rawText', 'title', 'description', 'dueDate', 'category', 'priority', 'checklist', 'tags'],
            },
          },
        });
        return response;
      } catch (err: any) {
        lastError = err;
        const status = err?.status;
        const isTransient =
          status === 503 || status === 429 || err?.name === 'AbortError' || err?.name === 'TimeoutError';
        if (!isTransient) {
          throw err;
        }
        await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
      }
    }
  }

  throw lastError;
}

app.post('/api/extract-reminder', async (req, res) => {
  try {
    const { image, prompt, ocrProfile, ocrText, captureMode } = req.body || {};
    const verifiedOcrText = typeof ocrText === 'string' ? ocrText.trim() : '';
    const validMode: LocalCaptureMode =
      captureMode === 'grocery' ||
      captureMode === 'receipt' ||
      captureMode === 'recipe' ||
      captureMode === 'fridge'
        ? captureMode
        : 'auto';

    if (!image && !verifiedOcrText && (!prompt || typeof prompt !== 'string' || !prompt.trim())) {
      return res.status(400).json({
        success: false,
        error: 'Please provide an image, OCR text, or prompt to extract reminder details.',
      });
    }

    const nowISO = new Date(Date.now() + 86400000).toISOString();
    const basePrompt =
      prompt ||
      (verifiedOcrText
        ? 'Convert this extracted note text into a structured reminder.'
        : 'Analyze this image (screenshot, flyer, bill, receipt, note, ticket, etc.) and extract reminder details.');

    const textWithOcr = verifiedOcrText
      ? `${basePrompt}\n\nExtracted text from note/image:\n"""\n${verifiedOcrText}\n"""`
      : basePrompt;

    const instructionText = `${textWithOcr}
Current time reference: ${new Date().toISOString()}.
Default dueDate if none is found in the content: ${nowISO}.
Extract clear, actionable reminder details as JSON.
- category must be one of: "Bills", "Events", "Tasks", "Travel", "Work", "Personal", "Health".
- priority must be one of: "high", "medium", "low".
- dueDate must be a valid ISO-8601 date-time string.`;

    // If user verified OCR text, structure via text-only AI; otherwise pass image directly
    let imageForVision: string | null = null;
    if (!verifiedOcrText && image && typeof image === 'string') {
      imageForVision = image;
    }

    // Prefer Groq Vision/Text LLM when configured AND not in OFFLINE_ONLY mode
    if (!isOfflineOnly() && process.env.GROQ_API_KEY) {
      try {
        const groqResult = await extractWithGroq(imageForVision, instructionText);
        if (groqResult) {
          return res.json({
            success: true,
            provider: 'groq',
            ocrProfile: ocrProfile || 'auto',
            rawText: typeof groqResult.rawText === 'string' ? groqResult.rawText.trim() : verifiedOcrText,
            data: groqResult,
          });
        }
      } catch (groqErr: any) {
        console.warn('Groq extraction failed, trying Gemini/local fallback:', groqErr?.message);
      }
    }

    if (!isOfflineOnly() && process.env.GEMINI_API_KEY) {
      try {
        let imagePart: { inlineData: { mimeType: string; data: string } } | null = null;
        if (imageForVision && typeof imageForVision === 'string') {
          const matches = imageForVision.match(/^data:(.+);base64,(.+)$/);
          if (matches) {
            imagePart = {
              inlineData: {
                mimeType: matches[1],
                data: matches[2],
              },
            };
          } else {
            imagePart = {
              inlineData: {
                mimeType: 'image/jpeg',
                data: imageForVision,
              },
            };
          }
        }

        const parts = [
          ...(imagePart ? [imagePart] : []),
          { text: instructionText },
        ];

        const response = await generateWithGeminiRetry(parts);
        const resultText = response.text;
        if (resultText) {
          const parsedData = JSON.parse(resultText);
          return res.json({
            success: true,
            provider: 'gemini',
            ocrProfile: ocrProfile || 'auto',
            rawText: typeof parsedData.rawText === 'string' ? parsedData.rawText.trim() : verifiedOcrText,
            data: parsedData,
          });
        }
      } catch (geminiErr: any) {
        console.warn('Gemini extraction failed, checking local fallback:', geminiErr?.message);
      }
    }

    // Offline / local Python OCR fallback when OCR text is available or Python OCR can read the image
    let localText = verifiedOcrText;
    let usedLocalOcr = false;
    let localResMeta: any = null;

    if (!localText && image && typeof image === 'string') {
      const localRes = await callPythonOcr({
        image,
        ocrProfile: (ocrProfile as OcrProfile) || 'auto',
        exportStages: true,
        captureMode: validMode,
      });
      localText = localRes.text || '';
      localResMeta = localRes;
      usedLocalOcr = true;
    }

    if (localText || (prompt && typeof prompt === 'string' && prompt.trim())) {
      const parsed = parseReminderLocally(
        localText || prompt.trim(),
        localText ? prompt : undefined,
        usedLocalOcr || ocrProfile === 'handwriting' ? ['handwritten'] : ['note'],
        validMode
      );
      return res.json({
        success: true,
        offlineOnly: isOfflineOnly(),
        provider: 'local-parser',
        ocrProfile: ocrProfile || 'auto',
        ocrText: localText,
        lines: localResMeta?.lines,
        imageWidth: localResMeta?.imageWidth,
        imageHeight: localResMeta?.imageHeight,
        exportDirName: localResMeta?.exportDir ? path.basename(localResMeta.exportDir) : undefined,
        needsUserPurpose: false,
        data: parsed,
      });
    }

    return res.json({
      success: true,
      offlineOnly: isOfflineOnly(),
      provider: 'local-parser',
      ocrProfile: ocrProfile || 'auto',
      ocrText: '',
      lines: localResMeta?.lines || [],
      imageWidth: localResMeta?.imageWidth,
      imageHeight: localResMeta?.imageHeight,
      exportDirName: localResMeta?.exportDir ? path.basename(localResMeta.exportDir) : undefined,
      needsUserPurpose: true,
      data: null,
    });
  } catch (error: any) {
    console.error('Error extracting reminder:', error);
    const status = typeof error?.status === 'number' ? error.status : 500;
    res.status(status >= 400 && status < 600 ? status : 500).json({
      success: false,
      error: error?.message || 'Failed to extract reminder details from image',
    });
  }
});

// Read-only OCR stage export listing & safe static serving
const OCR_EXPORTS_ROOT = path.resolve(process.cwd(), 'ocr-exports');

app.get('/api/ocr-exports', (_req, res) => {
  try {
    if (!fs.existsSync(OCR_EXPORTS_ROOT)) {
      return res.json({ success: true, root: OCR_EXPORTS_ROOT, runs: [] });
    }
    const runs = fs
      .readdirSync(OCR_EXPORTS_ROOT, { withFileTypes: true })
      .filter((d) => d.isDirectory() && d.name.startsWith('run-'))
      .map((d) => {
        const runPath = path.join(OCR_EXPORTS_ROOT, d.name);
        const files = fs
          .readdirSync(runPath)
          .filter((f) => f.endsWith('.png') || f.endsWith('.txt'))
          .sort();
        return {
          name: d.name,
          files,
          urls: files.map((f) => `/ocr-exports/${encodeURIComponent(d.name)}/${encodeURIComponent(f)}`),
        };
      })
      .sort((a, b) => b.name.localeCompare(a.name))
      .slice(0, 20);
    res.json({ success: true, root: OCR_EXPORTS_ROOT, runs });
  } catch (err: any) {
    res.status(500).json({ success: false, error: err?.message || 'Failed to list OCR exports.' });
  }
});

app.get('/ocr-exports/:run/:file', (req, res) => {
  const { run, file } = req.params;
  if (!/^run-[a-zA-Z0-9._-]+$/.test(run) || !/^[a-zA-Z0-9._-]+\.(png|txt)$/.test(file)) {
    return res.status(400).json({ success: false, error: 'Invalid export path.' });
  }
  const resolved = path.resolve(OCR_EXPORTS_ROOT, run, file);
  if (!resolved.startsWith(OCR_EXPORTS_ROOT + path.sep) || !fs.existsSync(resolved)) {
    return res.status(404).json({ success: false, error: 'Export file not found.' });
  }
  res.sendFile(resolved);
});

async function sendPushToTargets(
  targets: webpushType.PushSubscription[],
  payloadObj: { title: string; body: string; url?: string }
) {
  const payload = JSON.stringify({
    title: payloadObj.title,
    body: payloadObj.body,
    url: payloadObj.url || '/',
  });
  let sent = 0;
  let pruned = false;
  await Promise.all(
    targets.map(async (sub) => {
      try {
        await webpush.sendNotification(sub, payload);
        sent++;
      } catch (err: any) {
        if (err?.statusCode === 404 || err?.statusCode === 410) {
          pushSubscriptions.delete(sub.endpoint);
          pruned = true;
        } else {
          throw err;
        }
      }
    })
  );
  if (pruned) savePushSubscriptions();
  return sent;
}

// Web Push endpoints
app.post('/api/push/subscribe', (req, res) => {
  const { subscription } = req.body || {};
  if (!subscription?.endpoint || !subscription?.keys?.p256dh || !subscription?.keys?.auth) {
    return res.status(400).json({ success: false, error: 'Invalid push subscription payload.' });
  }
  pushSubscriptions.set(subscription.endpoint, subscription);
  savePushSubscriptions();
  res.json({ success: true });
});

app.post('/api/push/unsubscribe', (req, res) => {
  const { endpoint } = req.body || {};
  if (endpoint) {
    pushSubscriptions.delete(endpoint);
    savePushSubscriptions();
  }
  res.json({ success: true });
});

app.post('/api/push/send-test', async (req, res) => {
  try {
    if (!pushConfigured) {
      return res.status(500).json({
        success: false,
        error: 'Web Push VAPID keys are not configured on the server.',
      });
    }

    const { subscription, title, body } = req.body || {};
    const targets: webpushType.PushSubscription[] = subscription?.endpoint
      ? [subscription]
      : Array.from(pushSubscriptions.values());

    if (targets.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'No active push subscription found. Please enable notifications first.',
      });
    }

    const sent = await sendPushToTargets(targets, {
      title: title || 'SnapRemind Push Active',
      body: body || 'Web Push notifications are configured and working!',
      url: '/',
    });
    res.json({ success: true, sent });
  } catch (error: any) {
    console.error('Error sending push notification:', error);
    res.status(500).json({
      success: false,
      error: error?.message || 'Failed to send push notification.',
    });
  }
});

// Host-side active reminders sync & background random-window Web Push scheduler
interface SyncedReminderMeta {
  id: string;
  title: string;
  dueDate: string;
  completed: boolean;
  priority?: string;
}
const ACTIVE_REMINDERS_PATH = path.resolve(process.cwd(), 'data', 'active-reminders.json');
let syncedReminders: SyncedReminderMeta[] = [];
let lastBackgroundPushAt = 0;

function loadSyncedReminders() {
  try {
    if (fs.existsSync(ACTIVE_REMINDERS_PATH)) {
      const raw = JSON.parse(fs.readFileSync(ACTIVE_REMINDERS_PATH, 'utf8'));
      if (Array.isArray(raw)) syncedReminders = raw;
    }
  } catch {
    syncedReminders = [];
  }
}

loadSyncedReminders();

function getDueTodayReminders(): SyncedReminderMeta[] {
  const todayStr = new Date().toISOString().split('T')[0];
  const todayEndIso = `${todayStr}T23:59:59.999Z`;
  return syncedReminders.filter((r) => !r.completed && r.dueDate <= todayEndIso);
}

app.post('/api/reminders/sync', (req, res) => {
  const items = Array.isArray(req.body?.reminders) ? req.body.reminders : [];
  syncedReminders = items
    .filter((r: any) => r?.id && r?.title && r?.dueDate)
    .slice(0, 200)
    .map((r: any) => ({
      id: String(r.id),
      title: String(r.title).slice(0, 120),
      dueDate: String(r.dueDate),
      completed: Boolean(r.completed),
      priority: r.priority ? String(r.priority) : 'medium',
    }));
  try {
    fs.mkdirSync(path.dirname(ACTIVE_REMINDERS_PATH), { recursive: true });
    fs.writeFileSync(ACTIVE_REMINDERS_PATH, JSON.stringify(syncedReminders, null, 2), 'utf8');
  } catch (err) {
    console.warn('Failed to persist synced reminders:', err);
  }
  res.json({ success: true, count: syncedReminders.length, todayCount: getDueTodayReminders().length });
});

app.get('/api/reminders/today', (_req, res) => {
  res.json({ success: true, reminders: getDueTodayReminders() });
});

function startDailyPushScheduler() {
  const checkMs = Number(process.env.PUSH_NUDGE_INTERVAL_MS || 5 * 60 * 1000);
  const cooldownMs = Number(process.env.PUSH_NUDGE_COOLDOWN_MS || 2 * 60 * 60 * 1000);
  const timer = setInterval(async () => {
    if (!pushConfigured || pushSubscriptions.size === 0) return;
    if (Date.now() - lastBackgroundPushAt < cooldownMs) return;
    const due = getDueTodayReminders();
    if (due.length === 0) return;
    // Randomize firing probability (~40% per check window) so nudges feel organic
    if (Math.random() > 0.4) return;
    const pick = due[Math.floor(Math.random() * due.length)];
    try {
      const sent = await sendPushToTargets(Array.from(pushSubscriptions.values()), {
        title: "Today's Reminder",
        body: pick.title,
        url: '/',
      });
      if (sent > 0) lastBackgroundPushAt = Date.now();
    } catch (err) {
      console.warn('Background push scheduler error:', err);
    }
  }, checkMs);
  timer.unref?.();
}

app.post('/api/push/nudge', async (req, res) => {
  try {
    if (!pushConfigured) {
      return res.json({ success: false, skipped: true, reason: 'push-not-configured' });
    }
    const targets = Array.from(pushSubscriptions.values());
    if (targets.length === 0) {
      return res.json({ success: false, skipped: true, reason: 'no-subscriptions' });
    }
    const { title, body } = req.body || {};
    const sent = await sendPushToTargets(targets, {
      title: title || "Today's Reminder",
      body: body || 'You have an active reminder for today — tap to review.',
      url: '/',
    });
    res.json({ success: true, sent });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error?.message || 'Failed to send nudge.' });
  }
});

const isProd = process.env.NODE_ENV === 'production';

if (!isProd) {
  const vite = await createViteServer({
    server: { middlewareMode: true, allowedHosts: true },
    appType: 'spa',
  });
  app.use(vite.middlewares);
} else {
  const distDir = path.resolve(process.cwd(), 'dist');
  app.use(express.static(distDir));
  app.get('*', (_req, res) => {
    res.sendFile(path.join(distDir, 'index.html'));
  });
}

const port = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;
const host = process.env.HOST || '0.0.0.0';
const httpsKeyPath = process.env.HTTPS_KEY_PATH || '';
const httpsCertPath = process.env.HTTPS_CERT_PATH || '';
const useHttps = Boolean(
  httpsKeyPath && httpsCertPath && fs.existsSync(httpsKeyPath) && fs.existsSync(httpsCertPath)
);
const proto = useHttps ? 'https' : 'http';
const localUrl =
  process.env.APP_URL && !process.env.APP_URL.includes('MY_APP_URL')
    ? process.env.APP_URL
    : `${proto}://localhost:${port}`;
const cloudflareUrl = process.env.CLOUDFLARE_URL || '';

function getLanUrls(p: number, scheme: string): string[] {
  const urls: string[] = [];
  const ifaces = os.networkInterfaces();
  for (const entries of Object.values(ifaces)) {
    for (const entry of entries || []) {
      if (entry.family === 'IPv4' && !entry.internal) {
        urls.push(`${scheme}://${entry.address}:${p}`);
      }
    }
  }
  return urls;
}

const httpServer = useHttps
  ? https.createServer(
      { key: fs.readFileSync(httpsKeyPath), cert: fs.readFileSync(httpsCertPath) },
      app
    )
  : http.createServer(app);

httpServer.listen(port, host, () => {
  console.log('SnapRemind server running:');
  console.log(`  Local:      ${localUrl}`);
  console.log(`  Mode:       ${isOfflineOnly() ? 'OFFLINE_ONLY (100% in-house)' : 'Standard'}`);
  for (const lan of getLanUrls(port, proto)) {
    console.log(`  Phone LAN:  ${lan}`);
  }
  ensurePythonOcrSidecar()
    .then((ok) => {
      console.log(`  Python OCR: ${ok ? `ready (${PYTHON_OCR_URL})` : 'unavailable (503 until started)'}`);
    })
    .catch(() => {});
  startDailyPushScheduler();
  if (cloudflareUrl) {
    console.log(`  Cloudflare: ${cloudflareUrl}`);
  }

  if (!isOfflineOnly() && process.env.SKIP_AUTO_MODEL_DOWNLOAD !== '1') {
    ensureOfflineModels()
      .then((s) => {
        console.log(
          `Offline OCR models ready (handwriting=${s.handwritingModelState}, printed=${s.printedModelState}).`
        );
      })
      .catch((err) => {
        console.warn('Background offline OCR model provisioning error:', err?.message || err);
      });
  }
});
