import { execFile, spawn, type ChildProcess } from 'node:child_process';
import net from 'node:net';
import path from 'node:path';
import { promisify } from 'node:util';
import type {
  ExtractReminderHttpResult,
  LocalOcrHttpResult,
  ServerHandle,
} from './types';

const execFileAsync = promisify(execFile);

export function createAbortError(signal: AbortSignal, fallbackMessage: string): Error {
  const reason = signal.reason;
  const msg =
    reason instanceof Error
      ? reason.message
      : typeof reason === 'string' && reason.length > 0
        ? reason
        : fallbackMessage;
  const err = new Error(msg);
  err.name = 'AbortError';
  return err;
}

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

function isPidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    const code = (err as NodeJS.ErrnoException | undefined)?.code;
    return code === 'EPERM';
  }
}

async function countAndKillRemainingWin32Processes(
  pid: number | undefined,
  matcherSubstring?: string
): Promise<number> {
  const p = pid && pid > 0 ? pid : 0;
  const match = matcherSubstring ?? '';
  if (p === 0 && !match) return 0;

  const filterParts: string[] = [];
  if (match) filterParts.push("Name='chrome.exe'");
  if (p > 0) {
    filterParts.push(`ProcessId=${p}`, `ParentProcessId=${p}`);
  }
  const cimFilter = filterParts.join(' or ');

  const psScript = `
$p = [int]$env:SNAP2_KILL_PID
$m = $env:SNAP2_PROC_MATCH
$procs = @(Get-CimInstance Win32_Process -Filter "${cimFilter}" -ErrorAction SilentlyContinue | Where-Object {
  ($p -gt 0 -and ($_.ProcessId -eq $p -or $_.ParentProcessId -eq $p)) -or
  ($m -and $_.CommandLine -and $_.CommandLine.Contains($m))
})
foreach ($proc in $procs) {
  if ($proc.ProcessId -ne $PID) {
    try { Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue } catch {}
  }
}
Write-Output $procs.Count
`.trim();

  try {
    const { stdout } = await execFileAsync(
      'powershell.exe',
      ['-NoProfile', '-NonInteractive', '-Command', psScript],
      {
        windowsHide: true,
        timeout: 4000,
        env: {
          ...process.env,
          SNAP2_KILL_PID: String(p),
          SNAP2_PROC_MATCH: match,
        },
      }
    );
    const parsed = Number.parseInt(String(stdout).trim(), 10);
    return Number.isFinite(parsed) ? parsed : 0;
  } catch {
    return 0;
  }
}

export async function killProcessTree(
  pid: number | undefined,
  matcherSubstring?: string
): Promise<void> {
  if ((!pid || pid <= 0) && !matcherSubstring) return;
  try {
    if (process.platform === 'win32') {
      if (pid && pid > 0) {
        await execFileAsync('taskkill', ['/PID', String(pid), '/T', '/F'], {
          windowsHide: true,
          timeout: 4000,
        }).catch(() => {});
      }
    } else if (pid && pid > 0) {
      try {
        process.kill(-pid, 'SIGKILL');
      } catch {
        process.kill(pid, 'SIGKILL');
      }
    }
  } catch {
    // Process may already be exiting/exited
  }

  // Poll up to 6000ms until parent PID is dead AND (on Windows) zero matching Win32_Process children remain
  const deadline = Date.now() + 6000;
  while (Date.now() < deadline) {
    const parentAlive = pid && pid > 0 ? isPidAlive(pid) : false;
    if (process.platform === 'win32') {
      const remaining = await countAndKillRemainingWin32Processes(pid, matcherSubstring);
      if (!parentAlive && remaining === 0) return;
    } else if (!parentAlive) {
      return;
    }
    await new Promise((r) => setTimeout(r, 150));
  }
}

async function fetchJsonWithTimeout<T>(
  url: string,
  init: RequestInit,
  timeoutMs: number,
  externalSignal?: AbortSignal
): Promise<{ status: number; body: T }> {
  const controller = new AbortController();
  const onExternalAbort = () => controller.abort();
  if (externalSignal) {
    if (externalSignal.aborted) {
      controller.abort();
    } else {
      externalSignal.addEventListener('abort', onExternalAbort, { once: true });
    }
  }
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
    externalSignal?.removeEventListener('abort', onExternalAbort);
  }
}

function sleepAbortable(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(createAbortError(signal, 'Server startup aborted'));
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(createAbortError(signal!, 'Server startup aborted'));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

export async function startEphemeralServer(
  repoRoot: string,
  envOverrides: Record<string, string> = {},
  signal?: AbortSignal
): Promise<ServerHandle> {
  if (signal?.aborted) {
    throw createAbortError(signal, 'Server startup aborted');
  }
  const tsxCli = path.resolve(repoRoot, 'node_modules/tsx/dist/cli.mjs');
  let lastError: unknown;

  for (let attempt = 0; attempt < 2; attempt++) {
    if (signal?.aborted) {
      throw createAbortError(signal, 'Server startup aborted');
    }
    const port = await allocateFreePort();
    if (signal?.aborted) {
      throw createAbortError(signal, 'Server startup aborted');
    }
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
    let spawnError: unknown = null;
    child.on('error', (err) => {
      spawnError = err;
    });
    child.on('exit', (code) => {
      exitedEarly = true;
      exitCode = code;
    });

    try {
      const deadline = Date.now() + 30_000;
      let ready = false;
      while (Date.now() < deadline) {
        if (signal?.aborted) {
          throw createAbortError(signal, 'Server startup aborted');
        }
        if (spawnError) {
          const msg = spawnError instanceof Error ? spawnError.message : String(spawnError);
          throw new Error(`Failed to spawn Snap2 server: ${msg}`);
        }
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
            1500,
            signal
          );
          if (res.status === 200 && res.body?.status === 'ok') {
            ready = true;
            break;
          }
        } catch {
          if (signal?.aborted) {
            throw createAbortError(signal, 'Server startup aborted');
          }
          // Server still starting
        }
        await sleepAbortable(250, signal);
      }

      if (signal?.aborted) {
        throw createAbortError(signal, 'Server startup aborted');
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
