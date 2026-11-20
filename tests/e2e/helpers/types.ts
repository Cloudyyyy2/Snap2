export type TierId = 1 | 2 | 3 | 4;

export type FeatureId =
  | 'F1'
  | 'F2'
  | 'F3'
  | 'F4'
  | 'F5'
  | 'F6'
  | 'F7'
  | 'F8'
  | 'PAIRWISE'
  | 'SCENARIO';

export interface FixtureEntry {
  filename: string;
  groundTruth: string;
  expectedStyle: 'printed' | 'handwriting' | 'mixed' | 'blank' | 'noise';
  expectedLineCount: number;
  tags: string[];
  description: string;
}

export interface Manifest {
  version: number;
  generatedAt: string;
  fixtures: Record<string, FixtureEntry>;
}

export interface LocalOcrLine {
  text: string;
  confidence: number;
  bbox?: { x0: number; y0: number; x1: number; y1: number };
}

export interface LocalOcrHttpResult {
  success: boolean;
  error?: string;
  text: string;
  lines: LocalOcrLine[];
  profile: string;
  detectedStyle: 'printed' | 'handwriting' | 'mixed';
  recognizer: string;
  qualityScore: number;
  accuracyScore?: number;
  verifiedWords?: number;
  totalWords?: number;
  skewAngle?: number;
  darkInverted?: boolean;
  perspectiveWarped?: boolean;
  bordersCropped?: boolean;
  dpiScale?: number;
  appliedSteps?: string[];
  regionsCount?: number;
  processedPreview?: string;
  droppedHallucinations?: number;
  exportDir?: string;
}

export interface ExtractReminderHttpResult {
  success: boolean;
  error?: string;
  provider?: 'groq' | 'gemini' | 'local-parser' | string;
  ocrProfile?: string;
  rawText?: string;
  ocrText?: string;
  detectedStyle?: 'printed' | 'handwriting' | 'mixed';
  lines?: LocalOcrLine[];
  accuracyScore?: number;
  verifiedWords?: number;
  totalWords?: number;
  data?: {
    title?: string;
    category?: string;
    priority?: string;
    dueDate?: string;
    checklist?: Array<{ text: string; completed?: boolean }>;
    tags?: string[];
    notes?: string;
  };
}

export interface ServerHandle {
  port: number;
  baseUrl: string;
  pid: number | undefined;
  stop: () => Promise<void>;
  getHealth: () => Promise<{ status: number; body: Record<string, unknown> }>;
  getLocalOcrStatus: () => Promise<{ status: number; body: Record<string, unknown> }>;
  postLocalOcr: (payload: Record<string, unknown>, timeoutMs?: number) => Promise<{
    status: number;
    body: LocalOcrHttpResult;
  }>;
  postExtractReminder: (payload: Record<string, unknown>, timeoutMs?: number) => Promise<{
    status: number;
    body: ExtractReminderHttpResult;
  }>;
}

export interface InterceptedRequest {
  url: string;
  method: string;
  postData?: string;
  parsedBody?: Record<string, unknown>;
  status?: number;
  responseBody?: Record<string, unknown>;
}

export interface DomHarnessHandle {
  setAbortSignal: (signal?: AbortSignal) => void;
  resetPage: (baseUrl: string, signal?: AbortSignal) => Promise<void>;
  getUrl: () => Promise<string>;
  getPageText: () => Promise<string>;
  listButtons: () => Promise<Array<{ text: string; disabled: boolean; ariaLabel: string | null; className: string }>>;
  hasModeButton: (pattern: RegExp) => Promise<boolean>;
  clickModeButton: (pattern: RegExp) => Promise<string>;
  uploadFixture: (fixturePath: string) => Promise<void>;
  waitForScanComplete: (timeoutMs?: number, signal?: AbortSignal) => Promise<void>;
  waitForReviewSheet: (timeoutMs?: number, signal?: AbortSignal) => Promise<void>;
  getEditableText: () => Promise<string>;
  setEditableText: (value: string) => Promise<void>;
  getAccuracyBadge: () => Promise<{ found: boolean; rawText: string; percent: number | null }>;
  getDetectedStyleBadge: () => Promise<{ found: boolean; rawText: string; style: string | null }>;
  getLineConfidences: () => Promise<Array<{ text: string; confidencePct: number | null; raw: string }>>;
  clickCopyText: () => Promise<{ clicked: boolean; buttonText: string; feedbackText: string }>;
  readClipboardText: () => Promise<string>;
  clickConvertToReminder: (timeoutMs?: number, signal?: AbortSignal) => Promise<void>;
  clickSaveReminder: (timeoutMs?: number, signal?: AbortSignal) => Promise<void>;
  getStoredReminders: () => Promise<Array<Record<string, unknown>>>;
  getVisibleErrorBanner: () => Promise<string | null>;
  enableNetworkCapture: (options?: { failLocalOcrWithStatus?: number }) => Promise<void>;
  clearNetworkCapture: () => void;
  getCapturedRequests: () => InterceptedRequest[];
  close: () => Promise<void>;
}

export interface TestContext {
  repoRoot: string;
  fixturesDir: string;
  manifest: Manifest;
  signal: AbortSignal;
  getServer: () => Promise<ServerHandle>;
  spawnSecondaryServer: (envOverrides: Record<string, string>) => Promise<ServerHandle>;
  getDom: () => Promise<DomHarnessHandle>;
  createTempDir: (prefix?: string) => string;
  registerCleanup: (fn: () => Promise<void> | void) => void;
}

export interface TestCase {
  id: string;
  tier: TierId;
  feature: FeatureId;
  name: string;
  timeoutMs?: number;
  run: (ctx: TestContext) => Promise<void>;
}

export interface TestResultRecord {
  id: string;
  tier: TierId;
  feature: FeatureId;
  name: string;
  status: 'PASS' | 'FAIL';
  durationMs: number;
  error?: string;
}

export interface RunReport {
  timestamp: string;
  total: number;
  passed: number;
  failed: number;
  durationMs: number;
  byTier: Record<number, { total: number; passed: number; failed: number }>;
  results: TestResultRecord[];
}
