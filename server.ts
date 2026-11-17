import express from 'express';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import { createRequire } from 'module';
import type webpushType from 'web-push';
import dotenv from 'dotenv';
import path from 'path';
import {
  processImageWithLocalOcr,
  decodeBase64Image,
  isHandwritingModelAvailable,
  OcrProfile,
} from './server/handwritingRecognizer';
import { parseReminderLocally } from './server/localReminderParser';

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

// In-memory store for active browser push subscriptions in the current server session
const pushSubscriptions = new Map<string, webpushType.PushSubscription>();

app.get('/api/health', (_req, res) => {
  res.json({
    status: 'ok',
    aiConfigured: Boolean(process.env.GROQ_API_KEY || process.env.GEMINI_API_KEY),
    groqConfigured: Boolean(process.env.GROQ_API_KEY),
    geminiConfigured: Boolean(process.env.GEMINI_API_KEY),
    supabaseConfigured: Boolean(process.env.VITE_SUPABASE_URL && process.env.VITE_SUPABASE_ANON_KEY),
    pushConfigured,
  });
});

// Safe public client configuration (never exposes private keys or DB URLs)
app.get('/api/config', (_req, res) => {
  res.json({
    supabaseUrl: process.env.VITE_SUPABASE_URL || '',
    supabaseAnonKey: process.env.VITE_SUPABASE_ANON_KEY || '',
    vapidPublicKey,
    groqConfigured: Boolean(process.env.GROQ_API_KEY),
    pushConfigured,
  });
});

app.get('/api/local-ocr/status', (_req, res) => {
  res.json({
    handwritingModel: isHandwritingModelAvailable() ? 'ready' : 'missing',
    modelId: 'trocr-small-handwritten',
    groqConfigured: Boolean(process.env.GROQ_API_KEY),
    geminiConfigured: Boolean(process.env.GEMINI_API_KEY),
  });
});

app.post('/api/local-ocr', async (req, res) => {
  try {
    const { image, ocrProfile, exportStages } = req.body || {};
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

    const imageBuffer = decodeBase64Image(image);
    const result = await processImageWithLocalOcr({
      imageBuffer,
      ocrProfile: validProfile,
      exportStages: exportStages !== false,
    });

    res.json({
      success: true,
      ...result,
    });
  } catch (error: any) {
    console.error('Error running local OCR:', error);
    res.status(500).json({
      success: false,
      error: error?.message || 'Failed to process image with local OCR.',
    });
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
Return ONLY valid JSON matching this schema:
{
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
          content: 'You are SnapRemind, a precise assistant that extracts structured reminder JSON from screenshots and notes. Always respond with valid JSON only.',
        },
        {
          role: 'user',
          content: contentParts,
        },
      ],
      response_format: { type: 'json_object' },
      temperature: 0.2,
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
  const models = ['gemini-4-hydrogen', 'gemini-3.1-flash-lite'];
  let lastError: unknown;

  for (const model of models) {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const response = await ai.models.generateContent({
          model,
          contents: { parts },
          config: {
            abortSignal: AbortSignal.timeout(25000),
            thinkingConfig: {
              thinkingLevel: model === 'gemini-3.1-flash-lite' ? ThinkingLevel.MINIMAL : ThinkingLevel.LOW,
            },
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
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
              required: ['title', 'description', 'dueDate', 'category', 'priority', 'checklist', 'tags'],
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
    const { image, prompt, ocrProfile, ocrText } = req.body || {};
    const verifiedOcrText = typeof ocrText === 'string' ? ocrText.trim() : '';

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

    // If user verified local TrOCR text, structure via text-only AI (no need to re-send image) or fall back to offline parser
    const imageForVision = verifiedOcrText ? null : (image || null);

    // Prefer Groq Vision/Text LLM when configured, with Gemini fallback
    if (process.env.GROQ_API_KEY) {
      try {
        const groqResult = await extractWithGroq(imageForVision, instructionText);
        if (groqResult) {
          return res.json({
            success: true,
            provider: 'groq',
            ocrProfile: ocrProfile || 'auto',
            data: groqResult,
          });
        }
      } catch (groqErr: any) {
        console.warn('Groq extraction failed, trying Gemini/local fallback:', groqErr?.message);
      }
    }

    if (process.env.GEMINI_API_KEY) {
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
            data: parsedData,
          });
        }
      } catch (geminiErr: any) {
        console.warn('Gemini extraction failed, checking local fallback:', geminiErr?.message);
      }
    }

    // Offline / local deterministic fallback when OCR text is available or local TrOCR can read the image
    let localText = verifiedOcrText;
    let usedLocalOcr = false;

    if (!localText && image && typeof image === 'string' && isHandwritingModelAvailable()) {
      const localRes = await processImageWithLocalOcr({
        imageBuffer: decodeBase64Image(image),
        ocrProfile: ocrProfile === 'printed' ? 'handwriting' : (ocrProfile || 'handwriting'),
        exportStages: true,
      });
      localText = localRes.text;
      usedLocalOcr = true;
    }

    if (localText || (prompt && typeof prompt === 'string' && prompt.trim())) {
      const parsed = parseReminderLocally(
        localText || prompt.trim(),
        localText ? prompt : undefined,
        usedLocalOcr || ocrProfile === 'handwriting' ? ['handwritten'] : ['note']
      );
      return res.json({
        success: true,
        provider: 'local-parser',
        ocrProfile: ocrProfile || 'auto',
        ocrText: localText,
        data: parsed,
      });
    }

    return res.status(500).json({
      success: false,
      error: 'No AI provider succeeded and no readable local text was extracted.',
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

// Web Push endpoints
app.post('/api/push/subscribe', (req, res) => {
  const { subscription } = req.body || {};
  if (!subscription?.endpoint || !subscription?.keys?.p256dh || !subscription?.keys?.auth) {
    return res.status(400).json({ success: false, error: 'Invalid push subscription payload.' });
  }
  pushSubscriptions.set(subscription.endpoint, subscription);
  res.json({ success: true });
});

app.post('/api/push/unsubscribe', (req, res) => {
  const { endpoint } = req.body || {};
  if (endpoint) {
    pushSubscriptions.delete(endpoint);
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

    const payload = JSON.stringify({
      title: title || 'SnapRemind Push Active',
      body: body || 'Web Push notifications are configured and working!',
      url: '/',
    });

    await Promise.all(targets.map((sub) => webpush.sendNotification(sub, payload)));
    res.json({ success: true, sent: targets.length });
  } catch (error: any) {
    console.error('Error sending push notification:', error);
    res.status(500).json({
      success: false,
      error: error?.message || 'Failed to send push notification.',
    });
  }
});

const isProd = process.env.NODE_ENV === 'production';

if (!isProd) {
  const vite = await createViteServer({
    server: { middlewareMode: true },
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
app.listen(port, '0.0.0.0', () => {
  console.log(`SnapRemind server running at http://0.0.0.0:${port}`);
});
