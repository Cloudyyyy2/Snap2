import { execSync, spawn, type ChildProcess } from 'node:child_process';
import net from 'node:net';
import path from 'node:path';
import type {
  ExtractReminderHttpResult,
  LocalOcrHttpResult,
  ServerHandle,
} from './types';

export async function allocateFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address();
      if (!addr || typeof addr === 'string') {
        srv.close(() => reject(new Error('Failed to resolve ephemeral port')));
        return;
      }
      const port = addr.port;
      srv.close((err) => (err ? reject(err) : resolve(port)));
    });
  });
}

export async function killProcessTree(pid: number | undefined): Promise<void> {
  if (!pid || pid <= 0) return;
  try {
    if (process.platform === 'win32') {
      execSync(`taskkill /PID ${pid} /T /F`, { stdio: 'ignore' });
    } else {
      try {
        process.kill(-pid, 'SIGKILL');
      } catch {
        process.kill(pid, 'SIGKILL');
      }
    }
  } catch {
    // Process already exited
  }
}

async function fetchJsonWithTimeout<T>(
  url: string,
  init: RequestInit,
  timeoutMs: number
): Promise<{ status: number; body: T }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, { ...init, signal: controller.signal });
    const text = await resp.text();
    let body: T;
    try {
      body = JSON.parse(text) as T;
    } catch {
      body = { success: false, error: text } as unknown as T;
    }
    return { status: resp.status, body };
  } finally {
    clearTimeout(timer);
  }
}

export async function startEphemeralServer(
  repoRoot: string,
  envOverrides: Record<string, string> = {}
): Promise<ServerHandle> {
  const tsxCli = path.resolve(repoRoot, 'node_modules/tsx/dist/cli.mjs');
  let lastError: unknown;

  for (let attempt = 0; attempt < 2; attempt++) {
    const port = await allocateFreePort();
    const baseUrl = `http://127.0.0.1:${port}`;
    const logs: string[] = [];

    const child: ChildProcess = spawn(process.execPath, [tsxCli, 'server.ts'], {
      cwd: repoRoot,
      windowsHide: true,
      detached: process.platform !== 'win32',
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        PORT: String(port),
        APP_URL: baseUrl,
        NODE_ENV: 'development',
        DISABLE_HMR: 'true',
        SKIP_AUTO_MODEL_DOWNLOAD: '1',
        ...envOverrides,
      },
    });

    child.stdout?.on('data', (chunk) => {
      if (logs.length < 80) logs.push(String(chunk));
    });
    child.stderr?.on('data', (chunk) => {
      if (logs.length < 80) logs.push(String(chunk));
    });

    let exitedEarly = false;
    let exitCode: number | null = null;
    child.on('exit', (code) => {
      exitedEarly = true;
      exitCode = code;
    });

    try {
      const deadline = Date.now() + 30_000;
      let ready = false;
      while (Date.now() < deadline) {
        if (exitedEarly) {
          const logStr = logs.join('');
          if (logStr.includes('EADDRINUSE') && attempt === 0) {
            throw new Error('RETRY_EADDRINUSE');
          }
          throw new Error(
            `Server exited early (code=${exitCode}) before /api/health ready:\n${logStr}`
          );
        }
        try {
          const res = await fetchJsonWithTimeout<{ status?: string }>(
            `${baseUrl}/api/health`,
            { method: 'GET' },
            1500
          );
          if (res.status === 200 && res.body?.status === 'ok') {
            ready = true;
            break;
          }
        } catch {
          // Server still starting
        }
        await new Promise((r) => setTimeout(r, 250));
      }

      if (!ready) {
        throw new Error(`Timed out waiting for ${baseUrl}/api/health\nLogs:\n${logs.join('')}`);
      }

      return {
        port,
        baseUrl,
        pid: child.pid,
        stop: () => killProcessTree(child.pid),
        getHealth: () =>
          fetchJsonWithTimeout<Record<string, unknown>>(
            `${baseUrl}/api/health`,
            { method: 'GET' },
            10_000
          ),
        getLocalOcrStatus: () =>
          fetchJsonWithTimeout<Record<string, unknown>>(
            `${baseUrl}/api/local-ocr/status`,
            { method: 'GET' },
            10_000
          ),
        postLocalOcr: (payload, timeoutMs = 45_000) =>
          fetchJsonWithTimeout<LocalOcrHttpResult>(
            `${baseUrl}/api/local-ocr`,
            {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload),
            },
            timeoutMs
          ),
        postExtractReminder: (payload, timeoutMs = 45_000) =>
          fetchJsonWithTimeout<ExtractReminderHttpResult>(
            `${baseUrl}/api/extract-reminder`,
            {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload),
            },
            timeoutMs
          ),
      };
    } catch (err) {
      await killProcessTree(child.pid);
      lastError = err;
      if (err instanceof Error && err.message === 'RETRY_EADDRINUSE') {
        continue;
      }
      throw err;
    }
  }

  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}
