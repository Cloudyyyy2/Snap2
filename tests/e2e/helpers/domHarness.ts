import { spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import WebSocket from 'ws';
import { createAbortError, killProcessTree } from './serverHarness';
import type { DomHarnessHandle, InterceptedRequest } from './types';

function findChromeBinary(): string {
  const candidates: string[] = [];
  if (process.env.CHROME_PATH) {
    candidates.push(process.env.CHROME_PATH);
  }
  if (process.platform === 'win32') {
    candidates.push(
      'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
      'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe'
    );
    if (process.env.LOCALAPPDATA) {
      candidates.push(path.join(process.env.LOCALAPPDATA, 'Google\\Chrome\\Application\\chrome.exe'));
    }
  } else if (process.platform === 'darwin') {
    candidates.push('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome');
  } else {
    candidates.push('/usr/bin/google-chrome', '/usr/bin/chromium-browser', '/usr/bin/chromium');
  }

  for (const c of candidates) {
    if (c && fs.existsSync(c)) return c;
  }
  throw new Error(
    `Google Chrome binary not found. Checked: ${candidates.join(', ')}. Set CHROME_PATH if installed elsewhere.`
  );
}

class CdpConnection {
  private ws: WebSocket;
  private nextId = 1;
  private pending = new Map<
    number,
    { resolve: (res: Record<string, unknown>) => void; reject: (err: Error) => void; timer: NodeJS.Timeout }
  >();
  private eventHandlers = new Map<string, Set<(params: Record<string, unknown>, sessionId?: string) => void>>();

  constructor(ws: WebSocket) {
    this.ws = ws;
    ws.on('message', (raw: unknown) => {
      const msg = JSON.parse(String(raw)) as {
        id?: number;
        method?: string;
        params?: Record<string, unknown>;
        sessionId?: string;
        result?: Record<string, unknown>;
        error?: { message?: string; code?: number };
      };
      if (typeof msg.id === 'number') {
        const entry = this.pending.get(msg.id);
        if (entry) {
          this.pending.delete(msg.id);
          clearTimeout(entry.timer);
          if (msg.error) {
            entry.reject(new Error(`CDP error (${msg.error.code ?? ''}): ${msg.error.message ?? 'unknown'}`));
          } else {
            entry.resolve(msg.result ?? {});
          }
        }
      } else if (msg.method) {
        const handlers = this.eventHandlers.get(msg.method);
        if (handlers) {
          for (const h of handlers) {
            try {
              h(msg.params ?? {}, msg.sessionId);
            } catch {
              // ignore handler errors
            }
          }
        }
      }
    });
  }

  on(method: string, handler: (params: Record<string, unknown>, sessionId?: string) => void): () => void {
    let set = this.eventHandlers.get(method);
    if (!set) {
      set = new Set();
      this.eventHandlers.set(method, set);
    }
    set.add(handler);
    return () => set?.delete(handler);
  }

  send(
    method: string,
    params: Record<string, unknown> = {},
    sessionId?: string,
    timeoutMs = 15_000
  ): Promise<Record<string, unknown>> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`CDP command timed out (${timeoutMs}ms): ${method}`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      const payload: Record<string, unknown> = { id, method, params };
      if (sessionId) payload.sessionId = sessionId;
      this.ws.send(JSON.stringify(payload));
    });
  }

  close(): void {
    for (const [, entry] of this.pending) {
      clearTimeout(entry.timer);
      entry.reject(new Error('CDP connection closed'));
    }
    this.pending.clear();
    try {
      this.ws.close();
    } catch {
      try {
        (this.ws as unknown as { terminate?: () => void }).terminate?.();
      } catch {
        // ignore
      }
    }
  }
}

export async function launchDomHarness(signal?: AbortSignal): Promise<DomHarnessHandle> {
  if (signal?.aborted) {
    throw createAbortError(signal, 'Chrome launch aborted');
  }
  const chromePath = findChromeBinary();
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'snap2-chrome-e2e-'));
  const userDataDirBasename = path.basename(userDataDir);
  const stderrChunks: string[] = [];

  const chromeProc: ChildProcess = spawn(
    chromePath,
    [
      '--headless=new',
      '--disable-gpu',
      '--no-first-run',
      '--no-default-browser-check',
      '--disable-background-networking',
      '--disable-sync',
      '--remote-debugging-port=0',
      `--user-data-dir=${userDataDir}`,
      'about:blank',
    ],
    {
      windowsHide: true,
      detached: process.platform !== 'win32',
      stdio: ['ignore', 'ignore', 'pipe'],
    }
  );

  let chromeClosed = chromeProc.exitCode !== null || chromeProc.signalCode !== null;
  chromeProc.once('close', () => {
    chromeClosed = true;
  });

  let cdp: CdpConnection | null = null;
  let sessionId = '';
  let capturedRequests: InterceptedRequest[] = [];
  let fetchEnabled = false;
  let failLocalOcrWithStatus: number | undefined;
  let activeAbortSignal: AbortSignal | undefined;
  const networkIdToReq = new Map<string, InterceptedRequest>();
  const pendingPostDataByNetworkId = new Map<string, string>();
  let chromeLaunchError: unknown = null;
  let cleanupPromise: Promise<void> | null = null;

  chromeProc.on('error', (err) => {
    chromeLaunchError = err;
  });

  const cleanup = (): Promise<void> => {
    if (cleanupPromise) return cleanupPromise;
    cleanupPromise = (async () => {
      try {
        cdp?.close();
      } catch {
        // ignore
      }
      try {
        chromeProc.stderr?.destroy();
      } catch {
        // ignore
      }
      await killProcessTree(chromeProc.pid, userDataDirBasename);
      if (!chromeClosed && chromeProc.exitCode === null) {
        await new Promise<void>((resolve) => {
          const timer = setTimeout(resolve, 1000);
          chromeProc.once('close', () => {
            clearTimeout(timer);
            resolve();
          });
        });
      }
      const deadline = Date.now() + 10_000;
      while (fs.existsSync(userDataDir)) {
        try {
          await fs.promises.rm(userDataDir, { recursive: true, force: true });
          if (!fs.existsSync(userDataDir)) break;
        } catch {
          if (Date.now() >= deadline) break;
          await new Promise((r) => setTimeout(r, 200));
        }
      }
    })();
    return cleanupPromise;
  };

  const onStartupAbort = () => {
    try {
      cdp?.close();
    } catch {
      // ignore
    }
  };
  signal?.addEventListener('abort', onStartupAbort, { once: true });

  try {
    if (signal?.aborted) {
      throw createAbortError(signal, 'Chrome launch aborted');
    }

    const wsUrl = await new Promise<string>((resolve, reject) => {
      if (signal?.aborted) {
        reject(createAbortError(signal, 'Chrome launch aborted'));
        return;
      }
      if (chromeLaunchError) {
        const msg = chromeLaunchError instanceof Error ? chromeLaunchError.message : String(chromeLaunchError);
        reject(new Error(`Failed to launch Chrome: ${msg}`));
        return;
      }
      let settled = false;
      const cleanupListeners = () => {
        clearTimeout(timer);
        signal?.removeEventListener('abort', onAbortWs);
      };
      const onAbortWs = () => {
        if (settled) return;
        settled = true;
        cleanupListeners();
        reject(createAbortError(signal!, 'Chrome launch aborted while waiting for DevTools WS URL'));
      };
      signal?.addEventListener('abort', onAbortWs, { once: true });

      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        cleanupListeners();
        reject(new Error(`Timed out waiting for Chrome DevTools WS URL.\nStderr:\n${stderrChunks.join('')}`));
      }, 15_000);

      chromeProc.on('error', (err) => {
        if (settled) return;
        settled = true;
        cleanupListeners();
        reject(new Error(`Chrome process error: ${err.message}\nStderr:\n${stderrChunks.join('')}`));
      });

      chromeProc.on('exit', (code) => {
        if (settled) return;
        settled = true;
        cleanupListeners();
        reject(new Error(`Chrome exited prematurely (code=${code}).\nStderr:\n${stderrChunks.join('')}`));
      });

      chromeProc.stderr?.on('data', (chunk) => {
        const txt = String(chunk);
        stderrChunks.push(txt);
        const match = txt.match(/DevTools listening on (ws:\/\/[^\s]+)/i);
        if (match && !settled) {
          settled = true;
          cleanupListeners();
          resolve(match[1]);
        }
      });
    });

    if (signal?.aborted) {
      throw createAbortError(signal, 'Chrome launch aborted');
    }

    const ws = await new Promise<WebSocket>((resolve, reject) => {
      if (signal?.aborted) {
        reject(createAbortError(signal, 'Chrome launch aborted before WebSocket connect'));
        return;
      }
      const sock = new WebSocket(wsUrl);
      let settled = false;
      const cleanupSockListeners = () => {
        clearTimeout(timer);
        signal?.removeEventListener('abort', onAbortSock);
      };
      const onAbortSock = () => {
        if (settled) return;
        settled = true;
        cleanupSockListeners();
        try {
          (sock as unknown as { terminate?: () => void }).terminate?.();
        } catch {
          // ignore
        }
        reject(createAbortError(signal!, 'Chrome launch aborted during WebSocket connect'));
      };
      signal?.addEventListener('abort', onAbortSock, { once: true });

      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        cleanupSockListeners();
        try {
          (sock as unknown as { terminate?: () => void }).terminate?.();
        } catch {
          // ignore
        }
        reject(new Error('CDP WebSocket connect timeout'));
      }, 10_000);
      sock.once('open', () => {
        if (settled) return;
        settled = true;
        cleanupSockListeners();
        resolve(sock);
      });
      sock.once('error', (err: Error) => {
        if (settled) return;
        settled = true;
        cleanupSockListeners();
        reject(err);
      });
    });

    cdp = new CdpConnection(ws);
    if (signal?.aborted) {
      throw createAbortError(signal, 'Chrome launch aborted');
    }

    const targetsRes = await cdp.send('Target.getTargets');
    const targetInfos = (targetsRes.targetInfos as Array<{ targetId: string; type: string }>) ?? [];
    let pageTarget = targetInfos.find((t) => t.type === 'page');
    if (!pageTarget) {
      const created = await cdp.send('Target.createTarget', { url: 'about:blank' });
      pageTarget = { targetId: String(created.targetId), type: 'page' };
    }

    const attached = await cdp.send('Target.attachToTarget', {
      targetId: pageTarget.targetId,
      flatten: true,
    });
    sessionId = String(attached.sessionId);

    await cdp.send('Page.enable', {}, sessionId);
    await cdp.send('DOM.enable', {}, sessionId);
    await cdp.send('Runtime.enable', {}, sessionId);
    await cdp.send('Network.enable', {}, sessionId);

    if (signal?.aborted) {
      throw createAbortError(signal, 'Chrome launch aborted');
    }
    signal?.removeEventListener('abort', onStartupAbort);

    cdp.on('Fetch.requestPaused', async (params, sid) => {
      if (sid && sid !== sessionId) return;
      const requestId = String(params.requestId);
      const req = (params.request as {
        url?: string;
        method?: string;
        postData?: string;
        postDataEntries?: Array<{ bytes?: string }>;
      }) ?? {};
      const url = String(req.url ?? '');
      const method = String(req.method ?? 'GET');
      let postData = typeof req.postData === 'string' ? req.postData : undefined;
      if (!postData && Array.isArray(req.postDataEntries) && req.postDataEntries.length > 0) {
        const parts = req.postDataEntries
          .map((e) => (typeof e.bytes === 'string' ? Buffer.from(e.bytes, 'base64').toString('utf8') : ''))
          .join('');
        if (parts) postData = parts;
      }
      if (!postData && typeof params.networkId === 'string' && method === 'POST') {
        try {
          const pdRes = await cdp?.send(
            'Network.getRequestPostData',
            { requestId: params.networkId },
            sessionId,
            3000
          );
          if (pdRes && typeof pdRes.postData === 'string') {
            postData = pdRes.postData;
          }
        } catch {
          // ignore if unavailable
        }
      }
      let parsedBody: Record<string, unknown> | undefined;
      if (postData) {
        try {
          parsedBody = JSON.parse(postData) as Record<string, unknown>;
        } catch {
          // non-JSON
        }
      }

      const record: InterceptedRequest = { url, method, postData, parsedBody };
      capturedRequests.push(record);
      if (typeof params.networkId === 'string') {
        networkIdToReq.set(params.networkId, record);
        const pendingRaw = pendingPostDataByNetworkId.get(params.networkId);
        if (!record.parsedBody && pendingRaw) {
          record.postData = pendingRaw;
          try {
            record.parsedBody = JSON.parse(pendingRaw) as Record<string, unknown>;
          } catch {
            // non-JSON
          }
          pendingPostDataByNetworkId.delete(params.networkId);
        }
      }

      try {
        if (failLocalOcrWithStatus && url.includes('/api/local-ocr')) {
          record.status = failLocalOcrWithStatus;
          await cdp?.send(
            'Fetch.fulfillRequest',
            {
              requestId,
              responseCode: failLocalOcrWithStatus,
              responseHeaders: [{ name: 'Content-Type', value: 'application/json' }],
              body: Buffer.from(
                JSON.stringify({ success: false, error: 'Simulated server error for E2E test' }),
                'utf8'
              ).toString('base64'),
            },
            sessionId
          );
        } else {
          await cdp?.send('Fetch.continueRequest', { requestId }, sessionId);
        }
      } catch {
        // ignore if closed
      }
    });

    cdp.on('Network.requestWillBeSent', async (params, sid) => {
      if (sid && sid !== sessionId) return;
      const reqId = String(params.requestId ?? '');
      const req = (params.request as { url?: string; hasPostData?: boolean; postData?: string }) ?? {};
      const url = String(req.url ?? '');
      if (!url.includes('/api/local-ocr') && !url.includes('/api/extract-reminder')) return;
      const rec = networkIdToReq.get(reqId);
      if (rec && !rec.parsedBody) {
        let raw = typeof req.postData === 'string' ? req.postData : undefined;
        if (!raw && req.hasPostData) {
          try {
            const pdRes = await cdp?.send('Network.getRequestPostData', { requestId: reqId }, sessionId, 5000);
            if (pdRes && typeof pdRes.postData === 'string') raw = pdRes.postData;
          } catch {
            // ignore
          }
        }
        if (raw) {
          if (rec) {
            rec.postData = raw;
            try {
              rec.parsedBody = JSON.parse(raw) as Record<string, unknown>;
            } catch {
              // non-JSON
            }
          } else {
            pendingPostDataByNetworkId.set(reqId, raw);
          }
        }
      } else if (!rec) {
        let raw = typeof req.postData === 'string' ? req.postData : undefined;
        if (!raw && req.hasPostData) {
          try {
            const pdRes = await cdp?.send('Network.getRequestPostData', { requestId: reqId }, sessionId, 5000);
            if (pdRes && typeof pdRes.postData === 'string') raw = pdRes.postData;
          } catch {
            // ignore
          }
        }
        if (raw) pendingPostDataByNetworkId.set(reqId, raw);
      }
    });

    cdp.on('Network.responseReceived', (params, sid) => {
      if (sid && sid !== sessionId) return;
      const reqId = String(params.requestId ?? '');
      const resp = (params.response as { status?: number }) ?? {};
      const rec = networkIdToReq.get(reqId);
      if (rec && typeof resp.status === 'number') {
        rec.status = resp.status;
      }
    });

    cdp.on('Network.loadingFinished', async (params, sid) => {
      if (sid && sid !== sessionId) return;
      const reqId = String(params.requestId ?? '');
      const rec = networkIdToReq.get(reqId);
      if (!rec || (!rec.url.includes('/api/local-ocr') && !rec.url.includes('/api/extract-reminder'))) {
        return;
      }
      try {
        const bodyRes = await cdp?.send('Network.getResponseBody', { requestId: reqId }, sessionId, 4000);
        if (bodyRes && typeof bodyRes.body === 'string') {
          const raw = bodyRes.base64Encoded
            ? Buffer.from(bodyRes.body, 'base64').toString('utf8')
            : bodyRes.body;
          rec.responseBody = JSON.parse(raw) as Record<string, unknown>;
        }
      } catch {
        // ignore if body unavailable
      }
    });

    const evalPage = async <T>(expression: string): Promise<T> => {
      const res = await cdp!.send(
        'Runtime.evaluate',
        {
          expression,
          awaitPromise: true,
          returnByValue: true,
        },
        sessionId,
        20_000
      );
      if (res.exceptionDetails) {
        const details = res.exceptionDetails as { text?: string; exception?: { description?: string } };
        throw new Error(
          `Browser evaluation failed: ${details.exception?.description || details.text || JSON.stringify(details)}`
        );
      }
      const result = res.result as { value?: T } | undefined;
      return result?.value as T;
    };

    const waitForCondition = async <T>(
      fn: () => Promise<T | null | false>,
      timeoutMs: number,
      label: string,
      signal?: AbortSignal
    ): Promise<T> => {
      const effSignal = signal ?? activeAbortSignal;
      const throwIfAborted = () => {
        if (effSignal?.aborted) {
          const reason = effSignal.reason;
          throw reason instanceof Error ? reason : new Error(`Aborted while waiting for: ${label}`);
        }
      };
      const sleepAbortable = (ms: number) =>
        new Promise<void>((resolve, reject) => {
          if (effSignal?.aborted) {
            const reason = effSignal.reason;
            reject(reason instanceof Error ? reason : new Error(`Aborted while waiting for: ${label}`));
            return;
          }
          const onAbort = () => {
            clearTimeout(timer);
            const reason = effSignal?.reason;
            reject(reason instanceof Error ? reason : new Error(`Aborted while waiting for: ${label}`));
          };
          const timer = setTimeout(() => {
            effSignal?.removeEventListener('abort', onAbort);
            resolve();
          }, ms);
          effSignal?.addEventListener('abort', onAbort, { once: true });
        });

      const deadline = Date.now() + timeoutMs;
      while (Date.now() < deadline) {
        throwIfAborted();
        const val = await fn();
        throwIfAborted();
        if (val) return val;
        await sleepAbortable(150);
      }
      throw new Error(`Timed out after ${timeoutMs}ms waiting for: ${label}`);
    };

    const ensureStage1View = async (): Promise<void> => {
      const switched = await evalPage<boolean>(
        `(() => {
          const btns = Array.from(document.querySelectorAll('button')).filter(b => {
            const r = b.getBoundingClientRect();
            return r.width > 0 && r.height > 0;
          });
          const backBtn = btns.find(b => /Edit raw OCR text|Back to text/i.test((b.innerText || '').trim()));
          if (backBtn) {
            backBtn.click();
            return true;
          }
          return false;
        })()`
      );
      if (switched) {
        await new Promise((r) => setTimeout(r, 120));
      }
    };

    const handle: DomHarnessHandle = {
      setAbortSignal(signal?: AbortSignal): void {
        activeAbortSignal = signal;
      },

      async resetPage(baseUrl: string, signal?: AbortSignal): Promise<void> {
        const effSignal = signal ?? activeAbortSignal;
        if (effSignal?.aborted) {
          const reason = effSignal.reason;
          throw reason instanceof Error ? reason : new Error('Aborted before resetPage');
        }

        await cdp!.send('Page.navigate', { url: 'about:blank' }, sessionId);
        await new Promise((r) => setTimeout(r, 80));

        // Clear capture buffers AFTER navigating to about:blank so in-flight requests from prior page do not leak in
        capturedRequests = [];
        networkIdToReq.clear();
        pendingPostDataByNetworkId.clear();
        failLocalOcrWithStatus = undefined;

        await cdp!.send(
          'Browser.grantPermissions',
          {
            origin: baseUrl,
            permissions: ['clipboardReadWrite', 'clipboardSanitizedWrite'],
          },
          undefined
        );

        await cdp!.send('Page.navigate', { url: baseUrl }, sessionId);

        await waitForCondition(
          async () => {
            const ready = await evalPage<boolean>(
              `(() => {
                const text = document.body ? document.body.innerText : '';
                return document.readyState === 'complete' && text.includes('Live OCR Scanner');
              })()`
            );
            return ready ? true : null;
          },
          35_000,
          'OcrStudio Live OCR Scanner mount',
          effSignal
        );

        await evalPage<void>(
          `(() => {
            try {
              localStorage.clear();
              sessionStorage.clear();
            } catch {}
          })()`
        );
      },

      async getUrl(): Promise<string> {
        return evalPage<string>('window.location.href');
      },

      async getPageText(): Promise<string> {
        return evalPage<string>('document.body ? document.body.innerText : ""');
      },

      async listButtons() {
        return evalPage<Array<{ text: string; disabled: boolean; ariaLabel: string | null; className: string }>>(
          `Array.from(document.querySelectorAll('button')).filter(b => {
            const r = b.getBoundingClientRect();
            return r.width > 0 && r.height > 0;
          }).map(b => ({
            text: (b.innerText || b.textContent || '').trim(),
            disabled: Boolean(b.disabled),
            ariaLabel: b.getAttribute('aria-label'),
            className: typeof b.className === 'string' ? b.className : '',
          }))`
        );
      },

      async hasModeButton(pattern: RegExp): Promise<boolean> {
        const buttons = await handle.listButtons();
        return buttons.some((b) => pattern.test(b.text) || (b.ariaLabel ? pattern.test(b.ariaLabel) : false));
      },

      async clickModeButton(pattern: RegExp): Promise<string> {
        const source = pattern.source;
        const flags = pattern.flags.replace('g', '');
        const clicked = await evalPage<string | null>(
          `(() => {
            const re = new RegExp(${JSON.stringify(source)}, ${JSON.stringify(flags)});
            const btns = Array.from(document.querySelectorAll('button')).filter(b => {
              const r = b.getBoundingClientRect();
              return r.width > 0 && r.height > 0 && !b.disabled;
            });
            const target = btns.find(b => re.test((b.innerText || b.textContent || '').trim()) || (b.getAttribute('aria-label') && re.test(b.getAttribute('aria-label'))));
            if (!target) return null;
            target.click();
            return (target.innerText || target.textContent || '').trim();
          })()`
        );
        if (!clicked) {
          const all = await handle.listButtons();
          throw new Error(
            `Mode button matching ${pattern} not found. Visible buttons: ${JSON.stringify(all.map((b) => b.text))}`
          );
        }
        await new Promise((r) => setTimeout(r, 80));
        return clicked;
      },

      async uploadFixture(fixturePath: string): Promise<void> {
        const absPath = path.resolve(fixturePath);
        if (!fs.existsSync(absPath)) {
          throw new Error(`Fixture file not found for upload: ${absPath}`);
        }
        const doc = await cdp!.send('DOM.getDocument', { depth: -1 }, sessionId);
        const rootNodeId = (doc.root as { nodeId: number }).nodeId;
        const q = await cdp!.send(
          'DOM.querySelector',
          {
            nodeId: rootNodeId,
            selector: 'input[type="file"]:not([capture])',
          },
          sessionId
        );
        const nodeId = Number(q.nodeId ?? 0);
        if (!nodeId) {
          throw new Error('Could not find gallery file upload <input type="file"> element in DOM');
        }
        await cdp!.send('DOM.setFileInputFiles', { nodeId, files: [absPath] }, sessionId);
      },

      async waitForScanComplete(timeoutMs = 55_000, signal?: AbortSignal): Promise<void> {
        await waitForCondition(
          async () => {
            return evalPage<boolean>(
              `(() => {
                const bodyText = document.body ? document.body.innerText : '';
                const hasSpinner = /Extracting text|Recognizing|Scanning note|Running local OCR|Reading handwriting|Reading printed|Normalizing grayscale|Running offline OCR/i.test(bodyText);
                if (hasSpinner) return false;
                const textarea = document.querySelector('textarea');
                const hasConvertBtn = Array.from(document.querySelectorAll('button')).some(b => /Convert to Reminder|Save Reminder/i.test(b.innerText || ''));
                const hasError = /error|failed|could not|invalid|unavailable/i.test(bodyText) && Boolean(document.querySelector('[role="alert"], .text-red-500, .text-red-600, .text-rose-500, .bg-red-50, .bg-rose-50'));
                return Boolean((textarea && hasConvertBtn) || hasError);
              })()`
            );
          },
          timeoutMs,
          'OCR scan completion in browser',
          signal
        );
        await new Promise((r) => setTimeout(r, 150));
      },

      async waitForReviewSheet(timeoutMs = 15_000, signal?: AbortSignal): Promise<void> {
        await waitForCondition(
          async () =>
            evalPage<boolean>(
              `Boolean(document.querySelector('textarea') || Array.from(document.querySelectorAll('button')).some(b => /Convert to Reminder|Save Reminder/i.test(b.innerText || '')))`
            ),
          timeoutMs,
          'Review sheet visible',
          signal
        );
      },

      async getEditableText(): Promise<string> {
        await ensureStage1View();
        return evalPage<string>(
          `(() => {
            const ta = document.querySelector('textarea');
            return ta ? ta.value : '';
          })()`
        );
      },

      async setEditableText(value: string): Promise<void> {
        await ensureStage1View();
        await evalPage<void>(
          `(() => {
            const ta = document.querySelector('textarea');
            if (!ta) throw new Error('No textarea found to set editable text');
            const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value')?.set;
            if (setter) setter.call(ta, ${JSON.stringify(value)});
            else ta.value = ${JSON.stringify(value)};
            ta.dispatchEvent(new Event('input', { bubbles: true }));
            ta.dispatchEvent(new Event('change', { bubbles: true }));
          })()`
        );
        await new Promise((r) => setTimeout(r, 80));
      },

      async getAccuracyBadge() {
        await ensureStage1View();
        return evalPage<{ found: boolean; rawText: string; percent: number | null }>(
          `(() => {
            const spans = Array.from(document.querySelectorAll('span, div, p, badge'));
            const matchEl = spans.find(el => {
              const t = (el.innerText || '').trim();
              return t.length < 80 && /\\b(\\d{1,3})%\\s*(Accuracy|accuracy)/i.test(t) && el.children.length <= 3;
            });
            if (!matchEl) return { found: false, rawText: '', percent: null };
            const rawText = (matchEl.innerText || '').trim();
            const m = rawText.match(/(\\d{1,3})%/);
            return { found: true, rawText, percent: m ? Number(m[1]) : null };
          })()`
        );
      },

      async getDetectedStyleBadge() {
        await ensureStage1View();
        return evalPage<{ found: boolean; rawText: string; style: string | null }>(
          `(() => {
            const ta = document.querySelector('textarea');
            const root = ta ? (ta.closest('[role="dialog"], .fixed, .max-w-2xl, .max-w-xl, .space-y-4') || document.body) : document.body;
            const els = Array.from(root.querySelectorAll('span, div, p')).filter(el => el.tagName !== 'BUTTON' && !el.closest('button'));
            for (const el of els) {
              if (el.children.length > 2) continue;
              const t = (el.innerText || '').trim();
              if (!t || t.length > 60) continue;
              if (/detected\\s*style|style\\s*:|^handwriting$|^printed$|^mixed$/i.test(t)) {
                const m = t.match(/\\b(handwriting|printed|mixed)\\b/i);
                return { found: true, rawText: t, style: m ? m[1].toLowerCase() : null };
              }
            }
            return { found: false, rawText: '', style: null };
          })()`
        );
      },

      async getLineConfidences() {
        await ensureStage1View();
        return evalPage<Array<{ text: string; confidencePct: number | null; raw: string }>>(
          `(() => {
            const ta = document.querySelector('textarea');
            const root = ta ? (ta.closest('[role="dialog"], .fixed, .max-w-2xl, .max-w-xl, .space-y-4') || document.body) : document.body;
            const candidates = Array.from(root.querySelectorAll('li, [data-testid*="line"], [data-line-confidence], div')).filter(el => {
              if (el.tagName === 'BUTTON' || el.closest('button')) return false;
              const t = (el.innerText || '').trim();
              if (!t || t.length > 220) return false;
              if (/Accuracy|words verified|Live OCR|Offline Engine/i.test(t)) return false;
              return /\\b\\d{1,3}(?:\\.\\d+)?%(?!\\w)/.test(t) && el.children.length <= 4;
            });
            // Deduplicate nested elements keeping leaf-most matches
            const leaves = candidates.filter(el => !candidates.some(other => other !== el && el.contains(other)));
            return leaves.map(el => {
              const raw = (el.innerText || '').trim();
              const m = raw.match(/(\\d{1,3}(?:\\.\\d+)?)%/);
              const pct = m ? Number(m[1]) : null;
              const text = raw.replace(/\\b\\d{1,3}(?:\\.\\d+)?%(?!\\w)/g, '').replace(/[·|:\\-–]+/g, ' ').trim();
              return { text, confidencePct: pct, raw };
            });
          })()`
        );
      },

      async clickCopyText() {
        await ensureStage1View();
        const res = await evalPage<{ clicked: boolean; buttonText: string }>(
          `(() => {
            const btns = Array.from(document.querySelectorAll('button')).filter(b => {
              const r = b.getBoundingClientRect();
              return r.width > 0 && r.height > 0 && !b.disabled;
            });
            const btn = btns.find(b => /\\bcopy\\b/i.test((b.innerText || b.getAttribute('aria-label') || '').trim()));
            if (!btn) return { clicked: false, buttonText: '' };
            const buttonText = (btn.innerText || '').trim();
            btn.click();
            return { clicked: true, buttonText };
          })()`
        );
        await new Promise((r) => setTimeout(r, 180));
        const feedbackText = await evalPage<string>(
          `(() => {
            const btns = Array.from(document.querySelectorAll('button'));
            const copied = btns.find(b => /copied/i.test(b.innerText || ''));
            return copied ? (copied.innerText || '').trim() : '';
          })()`
        );
        return { ...res, feedbackText };
      },

      async readClipboardText(): Promise<string> {
        return evalPage<string>('navigator.clipboard.readText()');
      },

      async clickConvertToReminder(timeoutMs = 30_000, signal?: AbortSignal): Promise<void> {
        await ensureStage1View();
        const clicked = await evalPage<boolean>(
          `(() => {
            const btn = Array.from(document.querySelectorAll('button')).find(b => /Convert to Reminder/i.test((b.innerText || '').trim()));
            if (!btn) return false;
            btn.click();
            return true;
          })()`
        );
        if (!clicked) throw new Error('Convert to Reminder button not found');
        await waitForCondition(
          async () =>
            evalPage<boolean>(
              `Array.from(document.querySelectorAll('button')).some(b => /^Save Reminder$/i.test((b.innerText || '').trim()))`
            ),
          timeoutMs,
          'Save Reminder button after Convert to Reminder',
          signal
        );
      },

      async clickSaveReminder(timeoutMs = 15_000, signal?: AbortSignal): Promise<void> {
        const clicked = await evalPage<boolean>(
          `(() => {
            const btn = Array.from(document.querySelectorAll('button')).find(b => /^Save Reminder$/i.test((b.innerText || '').trim()));
            if (!btn) return false;
            btn.click();
            return true;
          })()`
        );
        if (!clicked) throw new Error('Save Reminder button not found');
        await waitForCondition(
          async () =>
            evalPage<boolean>(
              `!Array.from(document.querySelectorAll('button')).some(b => /^Save Reminder$/i.test((b.innerText || '').trim()))`
            ),
          timeoutMs,
          'Review sheet close after Save Reminder',
          signal
        );
      },

      async getStoredReminders(): Promise<Array<Record<string, unknown>>> {
        return evalPage<Array<Record<string, unknown>>>(
          `(() => {
            try {
              const raw = localStorage.getItem('snapremind_reminders_v1');
              return raw ? JSON.parse(raw) : [];
            } catch {
              return [];
            }
          })()`
        );
      },

      async getVisibleErrorBanner(): Promise<string | null> {
        return evalPage<string | null>(
          `(() => {
            const els = Array.from(document.querySelectorAll('[role="alert"], .text-red-500, .text-red-600, .text-red-700, .text-rose-500, .text-rose-600, .bg-red-50, .bg-rose-50'));
            const found = els.find(el => {
              const r = el.getBoundingClientRect();
              return r.width > 0 && r.height > 0 && (el.innerText || '').trim().length > 0;
            });
            return found ? (found.innerText || '').trim() : null;
          })()`
        );
      },

      async enableNetworkCapture(options?: { failLocalOcrWithStatus?: number }): Promise<void> {
        failLocalOcrWithStatus = options?.failLocalOcrWithStatus;
        if (!fetchEnabled) {
          await cdp!.send(
            'Fetch.enable',
            {
              patterns: [
                { urlPattern: '*/api/local-ocr*', requestStage: 'Request' },
                { urlPattern: '*/api/extract-reminder*', requestStage: 'Request' },
              ],
            },
            sessionId
          );
          fetchEnabled = true;
        }
      },

      clearNetworkCapture(): void {
        capturedRequests = [];
        networkIdToReq.clear();
        pendingPostDataByNetworkId.clear();
      },

      getCapturedRequests(): InterceptedRequest[] {
        return [...capturedRequests];
      },

      close: cleanup,
    };

    return handle;
  } catch (err) {
    signal?.removeEventListener('abort', onStartupAbort);
    await cleanup();
    throw err;
  }
}
