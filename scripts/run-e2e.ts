import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import { launchDomHarness } from '../tests/e2e/helpers/domHarness';
import { createAbortError, startEphemeralServer } from '../tests/e2e/helpers/serverHarness';
import type {
  DomHarnessHandle,
  Manifest,
  RunReport,
  ServerHandle,
  TestCase,
  TestContext,
  TestResultRecord,
  TierId,
} from '../tests/e2e/helpers/types';
import { tier1Tests } from '../tests/e2e/tier1-features.e2e';
import { tier2Tests } from '../tests/e2e/tier2-boundaries.e2e';
import { tier3Tests } from '../tests/e2e/tier3-combinations.e2e';
import { tier4Tests } from '../tests/e2e/tier4-scenarios.e2e';

interface CliOptions {
  tier: 'all' | TierId;
  dryRun: boolean;
  json: boolean;
  filter: string | null;
  timeoutOverrideMs: number | null;
}

function parseArgs(argv: string[]): CliOptions {
  const opts: CliOptions = {
    tier: 'all',
    dryRun: false,
    json: false,
    filter: null,
    timeoutOverrideMs: null,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--dry-run') {
      opts.dryRun = true;
    } else if (arg === '--json') {
      opts.json = true;
    } else if (arg === '--tier' && i + 1 < argv.length) {
      const val = argv[++i];
      if (val === 'all') {
        opts.tier = 'all';
      } else {
        const n = Number(val);
        if (n !== 1 && n !== 2 && n !== 3 && n !== 4) {
          throw new Error(`Invalid --tier value "${val}". Expected 1, 2, 3, 4, or all.`);
        }
        opts.tier = n as TierId;
      }
    } else if (arg.startsWith('--tier=')) {
      const val = arg.slice('--tier='.length);
      if (val === 'all') opts.tier = 'all';
      else {
        const n = Number(val);
        if (n !== 1 && n !== 2 && n !== 3 && n !== 4) {
          throw new Error(`Invalid --tier value "${val}". Expected 1, 2, 3, 4, or all.`);
        }
        opts.tier = n as TierId;
      }
    } else if (arg === '--filter' && i + 1 < argv.length) {
      opts.filter = argv[++i];
    } else if (arg.startsWith('--filter=')) {
      opts.filter = arg.slice('--filter='.length);
    } else if (arg === '--timeout' && i + 1 < argv.length) {
      const ms = Number(argv[++i]);
      if (!Number.isFinite(ms) || ms <= 0) throw new Error(`Invalid --timeout value "${ms}"`);
      opts.timeoutOverrideMs = ms;
    } else if (arg.startsWith('--timeout=')) {
      const ms = Number(arg.slice('--timeout='.length));
      if (!Number.isFinite(ms) || ms <= 0) throw new Error(`Invalid --timeout value "${ms}"`);
      opts.timeoutOverrideMs = ms;
    } else {
      throw new Error(`Unknown CLI flag: ${arg}`);
    }
  }
  return opts;
}

function findChromeBinaryForValidation(): string | null {
  const candidates: string[] = [];
  if (process.env.CHROME_PATH) candidates.push(process.env.CHROME_PATH);
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
  return candidates.find((c) => fs.existsSync(c)) ?? null;
}

async function runDryValidation(
  repoRoot: string,
  fixturesDir: string,
  manifest: Manifest,
  allTests: TestCase[],
  selectedTests: TestCase[],
  jsonMode: boolean
): Promise<void> {
  const errors: string[] = [];

  // 1. Validate unique test IDs and tier/feature counts
  const seenIds = new Set<string>();
  const tierCounts: Record<number, number> = { 1: 0, 2: 0, 3: 0, 4: 0 };
  const tier1Features: Record<string, number> = {};
  const tier2Features: Record<string, number> = {};

  for (const t of allTests) {
    if (!t.id || seenIds.has(t.id)) {
      errors.push(`Duplicate or empty test ID: ${t.id}`);
    }
    seenIds.add(t.id);
    tierCounts[t.tier] = (tierCounts[t.tier] ?? 0) + 1;
    if (t.tier === 1) tier1Features[t.feature] = (tier1Features[t.feature] ?? 0) + 1;
    if (t.tier === 2) tier2Features[t.feature] = (tier2Features[t.feature] ?? 0) + 1;
  }

  for (const fid of ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']) {
    if ((tier1Features[fid] ?? 0) < 5) {
      errors.push(`Tier 1 feature ${fid} has < 5 tests (${tier1Features[fid] ?? 0})`);
    }
    if ((tier2Features[fid] ?? 0) < 5) {
      errors.push(`Tier 2 feature ${fid} has < 5 tests (${tier2Features[fid] ?? 0})`);
    }
  }
  if (tierCounts[3] < 12) errors.push(`Tier 3 must have >= 12 tests (found ${tierCounts[3]})`);
  if (tierCounts[4] < 5) errors.push(`Tier 4 must have >= 5 scenarios (found ${tierCounts[4]})`);

  // 2. Validate manifest entries and decode every fixture PNG via sharp
  const fixtureEntries = Object.values(manifest.fixtures ?? {});
  if (fixtureEntries.length === 0) {
    errors.push('manifest.json contains 0 fixture entries');
  }
  for (const entry of fixtureEntries) {
    const fp = path.join(fixturesDir, entry.filename);
    if (!fs.existsSync(fp)) {
      errors.push(`Missing fixture PNG: ${fp}`);
      continue;
    }
    try {
      const meta = await sharp(fp).metadata();
      if (!meta.width || !meta.height) {
        errors.push(`Invalid PNG dimensions for ${entry.filename}`);
      }
    } catch (err) {
      errors.push(`sharp failed to decode ${entry.filename}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  // 3. Validate harness prerequisites
  const tsxCli = path.resolve(repoRoot, 'node_modules/tsx/dist/cli.mjs');
  const serverEntry = path.resolve(repoRoot, 'server.ts');
  if (!fs.existsSync(tsxCli)) errors.push(`Missing tsx CLI: ${tsxCli}`);
  if (!fs.existsSync(serverEntry)) errors.push(`Missing server entrypoint: ${serverEntry}`);
  const chromeBin = findChromeBinaryForValidation();
  if (!chromeBin) errors.push('Google Chrome binary not found for CDP UI harness');

  if (errors.length > 0) {
    if (jsonMode) {
      console.log(JSON.stringify({ ok: false, errors, totalTests: allTests.length, selectedTests: selectedTests.length }, null, 2));
    } else {
      console.error('Dry-run validation FAILED:');
      for (const e of errors) console.error(`  - ${e}`);
    }
    process.exit(1);
  }

  if (jsonMode) {
    console.log(
      JSON.stringify(
        {
          ok: true,
          dryRun: true,
          fixturesValidated: fixtureEntries.length,
          totalTests: allTests.length,
          selectedTests: selectedTests.length,
          tierCounts,
          chromeBinary: chromeBin,
        },
        null,
        2
      )
    );
  } else {
    console.log('E2E Dry-Run Validation PASSED:');
    console.log(`  Fixtures verified (sharp): ${fixtureEntries.length}`);
    console.log(
      `  Tests registered: ${allTests.length} (T1=${tierCounts[1]}, T2=${tierCounts[2]}, T3=${tierCounts[3]}, T4=${tierCounts[4]})`
    );
    console.log(`  Selected for current filter/tier: ${selectedTests.length}`);
    console.log(`  Chrome binary: ${chromeBin}`);
    console.log(`  Server entrypoint: ${serverEntry}`);
  }
}

async function main(): Promise<void> {
  const opts = parseArgs(process.argv.slice(2));
  const scriptDir = path.dirname(fileURLToPath(import.meta.url));
  const repoRoot = path.resolve(scriptDir, '..');
  const fixturesDir = path.resolve(repoRoot, 'tests/e2e/fixtures');
  const manifestPath = path.join(fixturesDir, 'manifest.json');

  if (!fs.existsSync(manifestPath)) {
    throw new Error(`Missing fixtures manifest at ${manifestPath}`);
  }
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8')) as Manifest;

  const allTests: TestCase[] = [...tier1Tests, ...tier2Tests, ...tier3Tests, ...tier4Tests];
  let selected = allTests.filter((t) => (opts.tier === 'all' ? true : t.tier === opts.tier));
  if (opts.filter !== null) {
    const q = opts.filter.toLowerCase();
    selected = selected.filter(
      (t) =>
        t.id.toLowerCase().includes(q) ||
        t.name.toLowerCase().includes(q) ||
        t.feature.toLowerCase().includes(q)
    );
  }

  if (opts.filter !== null && selected.length === 0 && !opts.dryRun) {
    const errMsg = `Filter "${opts.filter}" matched 0 tests (tier=${opts.tier}, totalRegistered=${allTests.length}).`;
    if (opts.json) {
      console.log(JSON.stringify({ ok: false, error: errMsg, total: 0, passed: 0, failed: 0 }, null, 2));
    } else {
      console.error(`Error: ${errMsg}`);
    }
    process.exit(1);
  }

  if (opts.dryRun) {
    await runDryValidation(repoRoot, fixturesDir, manifest, allTests, selected, opts.json);
    return;
  }

  const log = (msg: string) => {
    if (opts.json) process.stderr.write(msg + '\n');
    else console.log(msg);
  };

  let primaryServer: ServerHandle | null = null;
  let pendingServerPromise: Promise<ServerHandle> | null = null;
  let domHarness: DomHarnessHandle | null = null;
  let pendingDomPromise: Promise<DomHarnessHandle> | null = null;
  const secondaryServers: ServerHandle[] = [];
  const pendingSecondaryServers = new Set<Promise<ServerHandle>>();
  let activeTestController: AbortController | null = null;
  const tempDirs: string[] = [];
  const perTestCleanups: Array<() => Promise<void> | void> = [];
  const ocrExportsDir = path.join(repoRoot, 'ocr-exports');
  const preExistingExports = new Set(
    fs.existsSync(ocrExportsDir) ? fs.readdirSync(ocrExportsDir) : []
  );

  const settlePendingDomAndClose = async () => {
    const pDom = pendingDomPromise;
    if (pDom) {
      const maybeDom = await pDom.catch(() => null);
      if (maybeDom) {
        await maybeDom.close().catch(() => {});
      }
      if (domHarness === maybeDom) {
        domHarness = null;
      }
      if (pendingDomPromise === pDom) {
        pendingDomPromise = null;
      }
    }
    if (domHarness) {
      await domHarness.close().catch(() => {});
      domHarness = null;
    }
  };

  const settlePendingServerAndStop = async () => {
    const pSrv = pendingServerPromise;
    if (pSrv) {
      const maybeSrv = await pSrv.catch(() => null);
      if (maybeSrv) {
        await maybeSrv.stop().catch(() => {});
      }
      if (primaryServer === maybeSrv) {
        primaryServer = null;
      }
      if (pendingServerPromise === pSrv) {
        pendingServerPromise = null;
      }
    }
  };

  const settlePendingSecondaryServers = async () => {
    if (pendingSecondaryServers.size > 0) {
      const pending = Array.from(pendingSecondaryServers);
      pendingSecondaryServers.clear();
      const settled = await Promise.all(pending.map((p) => p.catch(() => null)));
      for (const s of settled) {
        if (s) await s.stop().catch(() => {});
      }
    }
    for (const s of secondaryServers.splice(0)) {
      await s.stop().catch(() => {});
    }
  };

  const shutdownAll = async () => {
    activeTestController?.abort(new Error('Runner shutting down'));
    for (const fn of perTestCleanups.splice(0)) {
      try {
        await fn();
      } catch {
        // ignore
      }
    }
    await settlePendingDomAndClose();
    await settlePendingSecondaryServers();
    await settlePendingServerAndStop();
    if (primaryServer) {
      await primaryServer.stop().catch(() => {});
      primaryServer = null;
    }
    for (const d of tempDirs.splice(0)) {
      try {
        fs.rmSync(d, { recursive: true, force: true });
      } catch {
        // ignore
      }
    }
    try {
      if (fs.existsSync(ocrExportsDir)) {
        for (const entry of fs.readdirSync(ocrExportsDir)) {
          if (!preExistingExports.has(entry)) {
            fs.rmSync(path.join(ocrExportsDir, entry), { recursive: true, force: true });
          }
        }
        if (fs.readdirSync(ocrExportsDir).length === 0) {
          fs.rmdirSync(ocrExportsDir);
        }
      }
    } catch {
      // ignore
    }
  };

  const onSigInt = () => {
    activeTestController?.abort(new Error('SIGINT'));
    shutdownAll().finally(() => process.exit(130));
  };
  process.once('SIGINT', onSigInt);

  const acquireServer = (signal?: AbortSignal): Promise<ServerHandle> => {
    if (signal?.aborted) {
      return Promise.reject(createAbortError(signal, 'Server startup aborted'));
    }
    if (primaryServer) return Promise.resolve(primaryServer);
    if (pendingServerPromise) return pendingServerPromise;

    log('[Harness] Starting isolated ephemeral Snap2 server...');
    const p = startEphemeralServer(repoRoot, {}, signal)
      .then(async (srv) => {
        if (signal?.aborted) {
          await srv.stop().catch(() => {});
          throw createAbortError(signal, 'Server startup aborted');
        }
        primaryServer = srv;
        log(`[Harness] Server ready at ${primaryServer.baseUrl} (pid=${primaryServer.pid})`);
        return srv;
      })
      .finally(() => {
        if (pendingServerPromise === p) {
          pendingServerPromise = null;
        }
      });
    pendingServerPromise = p;
    return p;
  };

  const acquireDom = (signal?: AbortSignal): Promise<DomHarnessHandle> => {
    if (signal?.aborted) {
      return Promise.reject(createAbortError(signal, 'Chrome launch aborted'));
    }
    if (domHarness) return Promise.resolve(domHarness);
    if (pendingDomPromise) return pendingDomPromise;

    log('[Harness] Launching headless Chrome CDP...');
    const p = launchDomHarness(signal)
      .then(async (dom) => {
        if (signal?.aborted) {
          await dom.close().catch(() => {});
          throw createAbortError(signal, 'Chrome launch aborted');
        }
        domHarness = dom;
        log('[Harness] Chrome CDP ready.');
        return dom;
      })
      .finally(() => {
        if (pendingDomPromise === p) {
          pendingDomPromise = null;
        }
      });
    pendingDomPromise = p;
    return p;
  };

  const baseCtx = {
    repoRoot,
    fixturesDir,
    manifest,
    getServer() {
      return acquireServer();
    },
    async spawnSecondaryServer(envOverrides: Record<string, string>, signal?: AbortSignal) {
      if (signal?.aborted) {
        throw createAbortError(signal, 'Secondary server startup aborted');
      }
      const p = startEphemeralServer(repoRoot, envOverrides, signal);
      pendingSecondaryServers.add(p);
      try {
        const srv = await p;
        if (signal?.aborted) {
          await srv.stop().catch(() => {});
          throw createAbortError(signal, 'Secondary server startup aborted');
        }
        secondaryServers.push(srv);
        return srv;
      } finally {
        pendingSecondaryServers.delete(p);
      }
    },
    getDom() {
      return acquireDom();
    },
    createTempDir(prefix = 'snap2-e2e-') {
      const d = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
      tempDirs.push(d);
      return d;
    },
    registerCleanup(fn: () => Promise<void> | void) {
      perTestCleanups.push(fn);
    },
  };

  const suiteStart = Date.now();
  const results: TestResultRecord[] = [];
  const byTier: Record<number, { total: number; passed: number; failed: number }> = {
    1: { total: 0, passed: 0, failed: 0 },
    2: { total: 0, passed: 0, failed: 0 },
    3: { total: 0, passed: 0, failed: 0 },
    4: { total: 0, passed: 0, failed: 0 },
  };

  try {
    for (const tc of selected) {
      const timeoutMs = opts.timeoutOverrideMs ?? tc.timeoutMs ?? 45_000;
      const t0 = Date.now();
      let status: 'PASS' | 'FAIL' = 'PASS';
      let errorMsg: string | undefined;

      const controller = new AbortController();
      activeTestController = controller;
      let testAcquiredDom = false;
      let timedOut = false;
      let timer: NodeJS.Timeout | undefined;

      const activeDomBefore = domHarness as DomHarnessHandle | null;
      if (activeDomBefore) {
        activeDomBefore.setAbortSignal(controller.signal);
      }

      const testCtx: TestContext = {
        ...baseCtx,
        signal: controller.signal,
        getServer() {
          return acquireServer(controller.signal);
        },
        spawnSecondaryServer(envOverrides: Record<string, string>) {
          return baseCtx.spawnSecondaryServer(envOverrides, controller.signal);
        },
        async getDom() {
          testAcquiredDom = true;
          const dom = await acquireDom(controller.signal);
          dom.setAbortSignal(controller.signal);
          return dom;
        },
      };

      const testPromise = tc.run(testCtx);
      // Prevent UnhandledPromiseRejection if timeout rejects first and testPromise rejects later
      testPromise.catch(() => {});

      const timeoutPromise = new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          timedOut = true;
          const timeoutErr = new Error(`Test timed out after ${timeoutMs}ms`);
          controller.abort(timeoutErr);
          reject(timeoutErr);
        }, timeoutMs);
      });

      try {
        await Promise.race([testPromise, timeoutPromise]);
      } catch (err) {
        status = 'FAIL';
        errorMsg = err instanceof Error ? err.message : String(err);
      } finally {
        if (timer) clearTimeout(timer);
        if (activeTestController === controller) {
          activeTestController = null;
        }
        const activeDomAfter = domHarness as DomHarnessHandle | null;
        if (activeDomAfter) {
          activeDomAfter.setAbortSignal(undefined);
        }
        if (timedOut) {
          if (pendingDomPromise || testAcquiredDom || activeDomAfter) {
            await settlePendingDomAndClose();
          }
          if (pendingServerPromise) {
            await settlePendingServerAndStop();
          }
        }
        for (const fn of perTestCleanups.splice(0)) {
          try {
            await fn();
          } catch {
            // ignore
          }
        }
        // Tear down secondary per-test servers (and any still-pending secondary launches) immediately
        await settlePendingSecondaryServers();
      }

      const durationMs = Date.now() - t0;
      results.push({
        id: tc.id,
        tier: tc.tier,
        feature: tc.feature,
        name: tc.name,
        status,
        durationMs,
        error: errorMsg,
      });

      byTier[tc.tier].total++;
      if (status === 'PASS') byTier[tc.tier].passed++;
      else byTier[tc.tier].failed++;

      log(`[Tier ${tc.tier}] ${tc.id} (${tc.name}) ... ${status} (${durationMs}ms)`);
      if (errorMsg) {
        log(`         -> ${errorMsg.split('\n')[0]}`);
      }
    }
  } finally {
    process.removeListener('SIGINT', onSigInt);
    await shutdownAll();
  }

  const passed = results.filter((r) => r.status === 'PASS').length;
  const failed = results.length - passed;
  const report: RunReport = {
    timestamp: new Date().toISOString(),
    total: results.length,
    passed,
    failed,
    durationMs: Date.now() - suiteStart,
    byTier,
    results,
  };

  if (opts.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log('\n=== E2E Summary ===');
    for (const tier of [1, 2, 3, 4]) {
      const st = byTier[tier];
      if (st.total > 0) {
        console.log(`Tier ${tier}: ${st.passed}/${st.total} passed (${st.failed} failed)`);
      }
    }
    console.log(`Total: ${passed}/${results.length} passed (${failed} failed) in ${(report.durationMs / 1000).toFixed(1)}s`);
  }

  process.exit(failed === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
