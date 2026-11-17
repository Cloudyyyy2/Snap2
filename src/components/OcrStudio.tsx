import React, { useState, useEffect, useCallback } from 'react';
import {
  RotateCcw,
  Sparkles,
  Loader2,
  X,
  Plus,
  CheckSquare,
  Square,
  ArrowLeft,
  Maximize2,
  FileText,
  Clock,
  Crop,
  CheckCircle2,
} from 'lucide-react';
import { CameraCapture } from './CameraCapture';
import { Reminder, Category, Priority, ChecklistItem, DetectedStyle } from '../types';
import { detectWordsOnImage, DetectedWordSquare } from '../lib/ocr/onDeviceWordDetector';

export interface RecentScanItem {
  id: string;
  image: string;
  previewText: string;
  engineLabel: string;
  createdAt: string;
}

interface OcrStudioProps {
  onSaveReminder: (reminder: Reminder) => Promise<void> | void;
  externalImage?: string | null;
  onClearExternalImage?: () => void;
  onSheetOpenChange?: (open: boolean) => void;
}

interface ScanExtractionMeta {
  linesCount: number;
  detectedStyle: DetectedStyle;
  engineLabel: string;
  appliedSteps?: string[];
  skewAngle?: number;
  regionsCount?: number;
  accuracyScore?: number;
  verifiedWords?: number;
  totalWords?: number;
}

interface DraftReminderState {
  title: string;
  description: string;
  dueDate: string; // datetime-local value or ISO converted
  category: Category;
  priority: Priority;
  checklist: ChecklistItem[];
  tags: string[];
  providerLabel: string;
}

const RECENT_SCANS_KEY = 'snapremind_recent_scans';
const CATEGORIES: Category[] = ['Bills', 'Events', 'Tasks', 'Travel', 'Work', 'Personal', 'Health'];
const PRIORITIES: Priority[] = ['high', 'medium', 'low'];

function getStoredRecentScans(): RecentScanItem[] {
  try {
    const raw = sessionStorage.getItem(RECENT_SCANS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return parsed;
    }
  } catch {
    // ignore storage errors
  }
  return [];
}

function saveStoredRecentScans(items: RecentScanItem[]) {
  try {
    sessionStorage.setItem(RECENT_SCANS_KEY, JSON.stringify(items.slice(0, 6)));
  } catch {
    // ignore quota errors
  }
}

export const OcrStudio: React.FC<OcrStudioProps> = ({
  onSaveReminder,
  externalImage = null,
  onClearExternalImage,
  onSheetOpenChange,
}) => {
  const [localModelReady, setLocalModelReady] = useState<boolean | null>(null);
  const [printedModelReady, setPrintedModelReady] = useState<boolean | null>(null);
  const [handwritingModelTier, setHandwritingModelTier] = useState<string | null>(null);
  const [printedModelTier, setPrintedModelTier] = useState<string | null>(null);

  const [sourceImage, setSourceImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [previewMode, setPreviewMode] = useState<'original' | 'processed'>('processed');
  const [reviewSheetOpen, setReviewSheetOpen] = useState(false);
  const [lightboxOpen, setLightboxOpen] = useState(false);

  const [isScanning, setIsScanning] = useState(false);
  const [statusStep, setStatusStep] = useState('');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [scanMeta, setScanMeta] = useState<ScanExtractionMeta | null>(null);
  const [editableText, setEditableText] = useState('');
  const [promptText, setPromptText] = useState('');
  const [draft, setDraft] = useState<DraftReminderState | null>(null);
  const [newStepText, setNewStepText] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [detectedWords, setDetectedWords] = useState<DetectedWordSquare[]>([]);
  const [showWordSquares, setShowWordSquares] = useState(false);

  const [recentScans, setRecentScans] = useState<RecentScanItem[]>(() => getStoredRecentScans());

  useEffect(() => {
    onSheetOpenChange?.(reviewSheetOpen);
  }, [reviewSheetOpen, onSheetOpenChange]);

  useEffect(() => {
    let active = true;
    fetch('/api/local-ocr/status')
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (active && data) {
          setLocalModelReady(data.handwritingModel === 'ready');
          setPrintedModelReady(data.printedModel === 'ready');
          setHandwritingModelTier(
            data.handwritingModelId === 'trocr-base-handwritten' ? 'TrOCR Base' : 'TrOCR'
          );
          setPrintedModelTier(
            data.printedModelId === 'tessdata_full' ? 'Tesseract Full' : 'Tesseract'
          );
        }
      })
      .catch(() => {
        if (active) {
          setLocalModelReady(false);
          setPrintedModelReady(false);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  const recordRecentScan = useCallback(
    (img: string, previewText: string, engineLabel: string) => {
      const nextItem: RecentScanItem = {
        id: `scan-${Date.now()}`,
        image: img,
        previewText: previewText.trim().slice(0, 80) || 'Scanned image',
        engineLabel,
        createdAt: new Date().toISOString(),
      };
      setRecentScans((prev) => {
        const filtered = prev.filter((item) => item.image !== img);
        const updated = [nextItem, ...filtered].slice(0, 6);
        saveStoredRecentScans(updated);
        return updated;
      });
    },
    []
  );

  const buildDraftFromApi = useCallback(
    (data: any, fallbackDescription: string, engineLabel: string): DraftReminderState => {
      const parsed = data.data || {};
      const validCategories: Category[] = [
        'Bills',
        'Events',
        'Tasks',
        'Travel',
        'Work',
        'Personal',
        'Health',
      ];
      const validPriorities: Priority[] = ['high', 'medium', 'low'];

      const category: Category = validCategories.includes(parsed.category)
        ? parsed.category
        : 'Tasks';
      const priority: Priority = validPriorities.includes(parsed.priority)
        ? parsed.priority
        : 'medium';

      const parsedDate = parsed.dueDate ? new Date(parsed.dueDate) : null;
      const isoDate =
        parsedDate && !isNaN(parsedDate.getTime())
          ? parsedDate.toISOString()
          : new Date(Date.now() + 86400000).toISOString();

      const checklist: ChecklistItem[] = Array.isArray(parsed.checklist)
        ? parsed.checklist
            .map((c: any, idx: number) => ({
              id: `chk-${Date.now()}-${idx}`,
              text: typeof c === 'string' ? c : c?.text || '',
              completed: Boolean(c?.completed),
            }))
            .filter((c: ChecklistItem) => c.text)
        : [];

      const tags: string[] = Array.isArray(parsed.tags) ? parsed.tags : ['ocr'];

      return {
        title: parsed.title || 'Scanned Reminder',
        description: parsed.description || fallbackDescription,
        dueDate: isoDate.slice(0, 16),
        category,
        priority,
        checklist,
        tags,
        providerLabel: engineLabel,
      };
    },
    []
  );

  const runExtractionPipeline = useCallback(
    async (image: string, options?: { forceVision?: boolean }) => {
      setSourceImage(image);
      setProcessedImage(null);
      setPreviewMode('processed');
      setReviewSheetOpen(true);
      setIsScanning(true);
      setErrorMsg(null);
      setDraft(null);
      setScanMeta(null);
      setDetectedWords([]);

      // Analyze word edges on device in parallel
      detectWordsOnImage(image, { maxDimension: 960, generateThumbnails: true, maxThumbnails: 24 })
        .then((res) => setDetectedWords(res.words))
        .catch(() => setDetectedWords([]));

      try {
        const canRunLocal =
          !options?.forceVision && (localModelReady !== false || printedModelReady !== false);

        if (canRunLocal) {
          setStatusStep('Auto-detecting printed & handwritten text offline...');

          const res = await fetch('/api/local-ocr', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              image,
              ocrProfile: 'auto',
              exportStages: true,
            }),
          });

          const localData = await res.json();
          if (!res.ok || !localData.success) {
            throw new Error(localData.error || `Offline OCR failed (HTTP ${res.status})`);
          }

          if (localData.processedPreview) {
            setProcessedImage(localData.processedPreview);
          }

          if (localData.recognizer !== 'vision-fallback') {
            const linesCount = Array.isArray(localData.lines) ? localData.lines.length : 0;
            const printedName = printedModelTier || 'Tesseract Full';
            const hwName = handwritingModelTier || 'TrOCR Base';
            const label =
              localData.recognizer === 'hybrid-host'
                ? `Auto · Mixed (${printedName} + ${hwName})`
                : localData.recognizer === 'tesseract-host'
                  ? `Auto · Printed (${printedName})`
                  : `Auto · Handwriting (${hwName})`;

            setEditableText(localData.text || '');
            setScanMeta({
              linesCount,
              detectedStyle: localData.detectedStyle || 'handwriting',
              engineLabel: label,
              appliedSteps: Array.isArray(localData.appliedSteps)
                ? localData.appliedSteps
                : undefined,
              skewAngle: localData.skewAngle,
              regionsCount: localData.regionsCount,
              accuracyScore: localData.accuracyScore,
              verifiedWords: localData.verifiedWords,
              totalWords: localData.totalWords,
            });
            recordRecentScan(image, localData.text || 'Scanned note', label);

            if (!localData.text?.trim()) {
              setErrorMsg(
                'No clear text lines were found offline. Edit the text below, tap Re-scan, or tap Try Vision AI.'
              );
            }
            return;
          }
        }

        // Manual Vision AI fallback ONLY when explicitly requested or when no local model is ready
        setStatusStep('Reading image with Vision AI fallback...');
        const res = await fetch('/api/extract-reminder', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image,
            prompt: promptText.trim() || undefined,
            ocrProfile: 'auto',
          }),
        });

        const contentType = res.headers.get('content-type') || '';
        const data = contentType.includes('application/json')
          ? await res.json()
          : { success: false, error: await res.text() };

        if (!res.ok || !data.success) {
          throw new Error(data.error || `Extraction failed (HTTP ${res.status})`);
        }

        const engineLabel =
          data.provider === 'local-parser' ? 'Offline · Local Parser' : 'Auto · Vision AI';

        const extractedTextPreview =
          data.rawText ||
          data.ocrText ||
          [data.data?.title, data.data?.description].filter(Boolean).join('\n');

        setEditableText(extractedTextPreview);
        setScanMeta({
          linesCount: extractedTextPreview ? extractedTextPreview.split('\n').length : 1,
          detectedStyle: 'printed',
          engineLabel,
        });
        setDraft(buildDraftFromApi(data, extractedTextPreview, engineLabel));
        recordRecentScan(image, data.data?.title || extractedTextPreview, engineLabel);
      } catch (err: any) {
        console.error('OCR scan error:', err);
        setErrorMsg(err?.message || 'Could not scan image. You can type or retry below.');
        setScanMeta({
          linesCount: 0,
          detectedStyle: 'printed',
          engineLabel: 'Manual Edit',
        });
      } finally {
        setIsScanning(false);
        setStatusStep('');
      }
    },
    [
      buildDraftFromApi,
      handwritingModelTier,
      localModelReady,
      printedModelReady,
      printedModelTier,
      promptText,
      recordRecentScan,
    ]
  );

  // Pick up workspace drag-and-drop image passed from App
  useEffect(() => {
    if (externalImage) {
      runExtractionPipeline(externalImage);
      onClearExternalImage?.();
    }
  }, [externalImage, onClearExternalImage, runExtractionPipeline]);

  // Global clipboard paste support on Scan tab
  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.startsWith('image/')) {
          const file = items[i].getAsFile();
          if (!file) continue;
          const reader = new FileReader();
          reader.onload = (ev) => {
            const result = ev.target?.result as string;
            if (result) {
              runExtractionPipeline(result);
            }
          };
          reader.readAsDataURL(file);
          break;
        }
      }
    };
    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, [runExtractionPipeline]);

  const readFileAndScan = useCallback(
    (file: File) => {
      const reader = new FileReader();
      reader.onload = (ev) => {
        const result = ev.target?.result as string;
        if (result) {
          runExtractionPipeline(result);
        }
      };
      reader.readAsDataURL(file);
    },
    [runExtractionPipeline]
  );

  const handleConvertTextToReminder = async () => {
    if (!editableText.trim() && !promptText.trim()) {
      setErrorMsg('Enter or scan some text before converting to a reminder.');
      return;
    }

    setIsScanning(true);
    setStatusStep('Structuring reminder...');
    setErrorMsg(null);

    try {
      // Final step only: pass verified extracted OCR text (never re-uploading image) to AI/local reminder structurer
      const res = await fetch('/api/extract-reminder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ocrText: editableText.trim(),
          prompt: promptText.trim() || undefined,
          ocrProfile: 'auto',
        }),
      });

      const contentType = res.headers.get('content-type') || '';
      const data = contentType.includes('application/json')
        ? await res.json()
        : { success: false, error: await res.text() };

      if (!res.ok || !data.success) {
        throw new Error(data.error || `Conversion failed (HTTP ${res.status})`);
      }

      const engineLabel = scanMeta?.engineLabel || 'OCR Studio';
      setDraft(buildDraftFromApi(data, editableText.trim(), engineLabel));
    } catch (err: any) {
      setErrorMsg(err?.message || 'Could not convert text to reminder.');
    } finally {
      setIsScanning(false);
      setStatusStep('');
    }
  };

  const handleSaveDraft = async () => {
    if (!draft) return;
    setIsSaving(true);
    try {
      const styleTags =
        scanMeta?.detectedStyle === 'handwriting'
          ? ['handwritten']
          : scanMeta?.detectedStyle === 'mixed'
            ? ['handwritten', 'ocr']
            : ['ocr'];

      const reminder: Reminder = {
        id: `rem-${Date.now()}`,
        title: draft.title.trim() || 'Scanned Reminder',
        description: draft.description.trim(),
        dueDate: draft.dueDate ? new Date(draft.dueDate).toISOString() : new Date().toISOString(),
        category: draft.category,
        priority: draft.priority,
        checklist: draft.checklist,
        tags: Array.from(new Set([...draft.tags, ...styleTags])),
        image: sourceImage || undefined,
        ocrText: editableText.trim() || undefined,
        ocrEngine: scanMeta?.engineLabel || draft.providerLabel,
        completed: false,
        starred: false,
        createdAt: new Date().toISOString(),
      };

      await onSaveReminder(reminder);
      setReviewSheetOpen(false);
      setDraft(null);
      setSourceImage(null);
      setEditableText('');
      setPromptText('');
      setScanMeta(null);
    } finally {
      setIsSaving(false);
    }
  };

  const applyPresetDueDate = (hoursFromNow: number, targetHour?: number) => {
    if (!draft) return;
    const d = new Date(Date.now() + hoursFromNow * 3600000);
    if (typeof targetHour === 'number') {
      d.setHours(targetHour, 0, 0, 0);
    }
    setDraft({ ...draft, dueDate: d.toISOString().slice(0, 16) });
  };

  const closeReviewSheet = () => {
    setReviewSheetOpen(false);
    setLightboxOpen(false);
  };

  const offlineStatusText =
    localModelReady === true || printedModelReady === true
      ? `Offline ${handwritingModelTier || 'TrOCR'} + ${printedModelTier || 'Tesseract'} · Auto-detect`
      : localModelReady === false
        ? 'Offline parser active'
        : 'Checking offline OCR...';

  return (
    <div className="w-full max-w-4xl mx-auto px-4 pt-3 pb-24 lg:px-8 lg:py-8 space-y-4">
      {/* Studio Intro Strip */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <h1 className="font-display text-lg sm:text-xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
            Live OCR Scanner
          </h1>
          <p className="text-xs sm:text-sm text-zinc-500 dark:text-zinc-400 mt-0.5">
            Automatically detects printed, handwritten, and mixed text offline.
          </p>
        </div>

        {sourceImage && !reviewSheetOpen && (
          <button
            type="button"
            onClick={() => setReviewSheetOpen(true)}
            className="min-h-[44px] px-3.5 rounded-xl bg-indigo-50 dark:bg-indigo-950/50 text-indigo-700 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800/60 text-xs font-medium whitespace-nowrap shrink-0"
          >
            Resume Scan
          </button>
        )}
      </div>

      {/* Main Feature: Always-Open Live Camera Viewfinder + Photo/Drop Fallback */}
      <CameraCapture
        embedded
        paused={reviewSheetOpen || lightboxOpen || isScanning}
        onCapture={(dataUrl) => runExtractionPipeline(dataUrl)}
        onUploadFile={readFileAndScan}
        statusNote={offlineStatusText}
      />

      {/* Recent Scans Strip */}
      {recentScans.length > 0 && (
        <section className="pt-2 space-y-2.5">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
              Recent scans
            </h2>
            <span className="font-mono-nums text-[11px] text-zinc-400 dark:text-zinc-500">
              {recentScans.length} saved in session
            </span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {recentScans.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => runExtractionPipeline(item.image)}
                className="min-h-[60px] p-2.5 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/60 hover:border-zinc-300 dark:hover:border-zinc-700 flex items-center gap-3 text-left transition-colors"
              >
                <img
                  src={item.image}
                  alt="Recent scan preview"
                  referrerPolicy="no-referrer"
                  className="w-11 h-11 rounded-xl object-cover border border-zinc-200 dark:border-zinc-800 shrink-0"
                />
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-medium text-zinc-900 dark:text-zinc-100 truncate">
                    {item.previewText}
                  </p>
                  <p className="text-[11px] text-zinc-500 dark:text-zinc-400 truncate mt-0.5">
                    {item.engineLabel}
                  </p>
                </div>
              </button>
            ))}
          </div>
        </section>
      )}

      {/* Slide-Up Bottom Sheet (Mobile) / Centered Sheet (Desktop) for OCR Review & Draft */}
      {reviewSheetOpen && (
        <div
          className="fixed inset-0 z-50 flex items-end md:items-center md:justify-center bg-zinc-950/60 dark:bg-black/75 backdrop-blur-[2px]"
          onClick={closeReviewSheet}
        >
          <div
            className="w-full md:max-w-xl bg-white dark:bg-zinc-900 border-t md:border border-zinc-200 dark:border-zinc-800 rounded-t-3xl md:rounded-2xl max-h-[92vh] flex flex-col overflow-hidden shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Mobile Drag Handle */}
            <div className="w-10 h-1.5 bg-zinc-300 dark:bg-zinc-700 rounded-full mx-auto mt-3 mb-1 md:hidden shrink-0" />

            {/* Sheet Header with Zoomable Source / Processed Thumbnail */}
            <div className="flex items-center justify-between gap-3 px-4 sm:px-5 py-3 border-b border-zinc-200 dark:border-zinc-800 shrink-0">
              <div className="flex items-center gap-3 min-w-0">
                {sourceImage && (
                  <button
                    type="button"
                    onClick={() => setLightboxOpen(true)}
                    className="relative w-12 h-12 rounded-xl overflow-hidden border border-zinc-200 dark:border-zinc-800 bg-zinc-100 dark:bg-zinc-950 shrink-0 group"
                    aria-label="Zoom scanned image"
                  >
                    <img
                      src={
                        previewMode === 'processed' && processedImage
                          ? processedImage
                          : sourceImage
                      }
                      alt="Scanned note thumbnail"
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-cover"
                    />
                    <span className="absolute bottom-0.5 right-0.5 p-0.5 rounded bg-black/60 text-white">
                      <Maximize2 className="w-2.5 h-2.5" />
                    </span>
                  </button>
                )}

                <div className="min-w-0">
                  <h2 className="font-display text-sm sm:text-base font-semibold text-zinc-900 dark:text-zinc-100 truncate">
                    {draft ? 'Review Reminder' : 'Extracted Text'}
                  </h2>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400 font-mono-nums truncate">
                    {isScanning
                      ? statusStep || 'Processing...'
                      : scanMeta
                        ? `${scanMeta.linesCount} ${scanMeta.linesCount === 1 ? 'line' : 'lines'} · ${scanMeta.engineLabel}`
                        : 'Ready to review'}
                  </p>
                </div>
              </div>

              <button
                type="button"
                onClick={closeReviewSheet}
                className="min-h-[44px] min-w-[44px] flex items-center justify-center rounded-xl text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 hover:bg-zinc-100 dark:hover:bg-zinc-800 shrink-0"
                aria-label="Close review sheet"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Scrollable Body */}
            <div className="flex-1 overflow-y-auto p-4 sm:p-5 space-y-4">
              {errorMsg && (
                <div className="p-3 bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800/60 rounded-xl text-xs text-rose-700 dark:text-rose-300 leading-relaxed">
                  {errorMsg}
                </div>
              )}

              {isScanning ? (
                <div className="py-10 flex flex-col items-center justify-center gap-3 text-center">
                  <Loader2 className="w-6 h-6 animate-spin text-indigo-600 dark:text-indigo-400" />
                  <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200">
                    {statusStep || 'Reading note...'}
                  </p>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">
                    Cleaning shadows, removing notebook lines, and extracting text
                  </p>
                </div>
              ) : draft ? (
                /* Stage 2: Editable Reminder Draft */
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <button
                      type="button"
                      onClick={() => setDraft(null)}
                      className="min-h-[36px] text-xs font-medium text-indigo-600 dark:text-indigo-400 hover:underline inline-flex items-center gap-1"
                    >
                      <ArrowLeft className="w-3.5 h-3.5" />
                      <span>Edit raw OCR text</span>
                    </button>
                    <span className="text-[11px] text-zinc-500 dark:text-zinc-400">
                      {draft.providerLabel}
                    </span>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-zinc-600 dark:text-zinc-400 mb-1.5">
                      Title
                    </label>
                    <input
                      type="text"
                      value={draft.title}
                      onChange={(e) => setDraft({ ...draft, title: e.target.value })}
                      className="w-full min-h-[44px] px-3 bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-xl text-base sm:text-sm text-zinc-900 dark:text-zinc-100 focus:outline-none focus:border-indigo-500"
                    />
                  </div>

                  {/* Quick Due Date Tap Presets + Picker */}
                  <div className="space-y-2">
                    <label className="block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                      Due Date & Time
                    </label>
                    <div className="grid grid-cols-3 gap-2">
                      <button
                        type="button"
                        onClick={() => applyPresetDueDate(0, 18)}
                        className="min-h-[40px] px-2 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 hover:bg-zinc-100 dark:hover:bg-zinc-800 text-xs font-medium text-zinc-700 dark:text-zinc-300 inline-flex items-center justify-center gap-1 whitespace-nowrap"
                      >
                        <Clock className="w-3 h-3" />
                        <span>Today 6 PM</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => applyPresetDueDate(24, 9)}
                        className="min-h-[40px] px-2 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 hover:bg-zinc-100 dark:hover:bg-zinc-800 text-xs font-medium text-zinc-700 dark:text-zinc-300 inline-flex items-center justify-center whitespace-nowrap"
                      >
                        Tomorrow 9 AM
                      </button>
                      <button
                        type="button"
                        onClick={() => applyPresetDueDate(168, 9)}
                        className="min-h-[40px] px-2 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 hover:bg-zinc-100 dark:hover:bg-zinc-800 text-xs font-medium text-zinc-700 dark:text-zinc-300 inline-flex items-center justify-center whitespace-nowrap"
                      >
                        Next Week
                      </button>
                    </div>
                    <input
                      type="datetime-local"
                      value={draft.dueDate}
                      onChange={(e) => setDraft({ ...draft, dueDate: e.target.value })}
                      className="w-full min-h-[44px] px-3 font-mono-nums bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-xl text-base sm:text-xs text-zinc-800 dark:text-zinc-200 focus:outline-none focus:border-indigo-500"
                    />
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs font-medium text-zinc-600 dark:text-zinc-400 mb-1.5">
                        Category
                      </label>
                      <select
                        value={draft.category}
                        onChange={(e) =>
                          setDraft({ ...draft, category: e.target.value as Category })
                        }
                        className="w-full min-h-[44px] px-3 bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-xl text-base sm:text-xs font-medium text-zinc-800 dark:text-zinc-200 focus:outline-none focus:border-indigo-500"
                      >
                        {CATEGORIES.map((cat) => (
                          <option key={cat} value={cat}>
                            {cat}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-zinc-600 dark:text-zinc-400 mb-1.5">
                        Priority
                      </label>
                      <div className="grid grid-cols-3 gap-1 p-1 min-h-[44px] bg-zinc-100 dark:bg-zinc-950 rounded-xl border border-zinc-200 dark:border-zinc-800">
                        {PRIORITIES.map((p) => (
                          <button
                            key={p}
                            type="button"
                            onClick={() => setDraft({ ...draft, priority: p })}
                            className={`text-xs font-medium rounded-lg capitalize transition-colors ${
                              draft.priority === p
                                ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-sm'
                                : 'text-zinc-500 dark:text-zinc-400'
                            }`}
                          >
                            {p}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-zinc-600 dark:text-zinc-400 mb-1.5">
                      Notes
                    </label>
                    <textarea
                      rows={3}
                      value={draft.description}
                      onChange={(e) => setDraft({ ...draft, description: e.target.value })}
                      className="w-full p-3 bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-xl text-base sm:text-xs text-zinc-900 dark:text-zinc-100 focus:outline-none focus:border-indigo-500 leading-relaxed"
                    />
                  </div>

                  {/* Checklist Action Steps */}
                  <div className="space-y-2">
                    <label className="block text-xs font-medium text-zinc-600 dark:text-zinc-400">
                      Action Steps
                    </label>
                    {draft.checklist.length > 0 && (
                      <div className="divide-y divide-zinc-100 dark:divide-zinc-800 border border-zinc-200 dark:border-zinc-800 rounded-xl overflow-hidden">
                        {draft.checklist.map((item) => (
                          <div
                            key={item.id}
                            className="flex items-center justify-between px-3 py-2 bg-zinc-50/70 dark:bg-zinc-950/60"
                          >
                            <button
                              type="button"
                              onClick={() =>
                                setDraft({
                                  ...draft,
                                  checklist: draft.checklist.map((c) =>
                                    c.id === item.id ? { ...c, completed: !c.completed } : c
                                  ),
                                })
                              }
                              className="flex items-center gap-2.5 text-left flex-1 min-w-0 min-h-[36px]"
                            >
                              {item.completed ? (
                                <CheckSquare className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
                              ) : (
                                <Square className="w-4 h-4 text-zinc-400 shrink-0" />
                              )}
                              <span className="text-xs text-zinc-800 dark:text-zinc-200 truncate">
                                {item.text}
                              </span>
                            </button>
                            <button
                              type="button"
                              onClick={() =>
                                setDraft({
                                  ...draft,
                                  checklist: draft.checklist.filter((c) => c.id !== item.id),
                                })
                              }
                              className="p-2 text-zinc-400 hover:text-rose-500"
                              aria-label="Remove step"
                            >
                              <X className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={newStepText}
                        onChange={(e) => setNewStepText(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && newStepText.trim()) {
                            e.preventDefault();
                            setDraft({
                              ...draft,
                              checklist: [
                                ...draft.checklist,
                                {
                                  id: `chk-${Date.now()}`,
                                  text: newStepText.trim(),
                                  completed: false,
                                },
                              ],
                            });
                            setNewStepText('');
                          }
                        }}
                        placeholder="Add an action step..."
                        className="flex-1 min-h-[44px] px-3 bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-xl text-base sm:text-xs text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 focus:outline-none focus:border-indigo-500"
                      />
                      <button
                        type="button"
                        onClick={() => {
                          if (!newStepText.trim()) return;
                          setDraft({
                            ...draft,
                            checklist: [
                              ...draft.checklist,
                              {
                                id: `chk-${Date.now()}`,
                                text: newStepText.trim(),
                                completed: false,
                              },
                            ],
                          });
                          setNewStepText('');
                        }}
                        className="min-h-[44px] px-3.5 bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 text-xs font-medium rounded-xl inline-flex items-center gap-1 whitespace-nowrap shrink-0"
                      >
                        <Plus className="w-3.5 h-3.5" />
                        <span>Add</span>
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                /* Stage 1: Editable Extracted OCR Text */
                <div className="space-y-3.5">
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between gap-2">
                      <label className="text-xs font-medium text-zinc-700 dark:text-zinc-300 inline-flex items-center gap-1.5">
                        <FileText className="w-3.5 h-3.5 text-zinc-400" />
                        <span>Editable extracted text</span>
                      </label>
                      {processedImage && (
                        <div className="inline-flex items-center gap-1 p-0.5 rounded-lg bg-zinc-100 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800">
                          <button
                            type="button"
                            onClick={() => setPreviewMode('processed')}
                            className={`px-2 py-1 text-[11px] font-medium rounded-md transition-colors ${
                              previewMode === 'processed'
                                ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-sm'
                                : 'text-zinc-500 dark:text-zinc-400'
                            }`}
                          >
                            Cleaned
                          </button>
                          <button
                            type="button"
                            onClick={() => setPreviewMode('original')}
                            className={`px-2 py-1 text-[11px] font-medium rounded-md transition-colors ${
                              previewMode === 'original'
                                ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-sm'
                                : 'text-zinc-500 dark:text-zinc-400'
                            }`}
                          >
                            Original
                          </button>
                        </div>
                      )}
                    </div>
                    <div className="flex items-center justify-between gap-2 flex-wrap">
                      {scanMeta?.appliedSteps && scanMeta.appliedSteps.length > 0 && (
                        <p className="text-[11px] text-zinc-500 dark:text-zinc-400 font-mono-nums truncate">
                          {scanMeta.appliedSteps.slice(0, 5).join(' · ')}
                        </p>
                      )}
                      {scanMeta?.accuracyScore !== undefined && (
                        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-emerald-50 dark:bg-emerald-950/50 border border-emerald-500/30 text-emerald-700 dark:text-emerald-300 text-[11px] font-medium">
                          <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                          <span>
                            {scanMeta.accuracyScore}% Accuracy · {scanMeta.verifiedWords ?? 0} words verified
                          </span>
                        </span>
                      )}
                    </div>
                    <textarea
                      rows={5}
                      value={editableText}
                      onChange={(e) => setEditableText(e.target.value)}
                      placeholder="Review or edit recognized text before converting..."
                      className="w-full p-3.5 bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-xl text-base sm:text-sm text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 focus:outline-none focus:border-indigo-500 leading-relaxed"
                    />

                    {/* On-device Detected Word Squares inspector strip */}
                    {detectedWords.length > 0 && (
                      <div className="space-y-1.5 pt-1">
                        <div className="flex items-center justify-between">
                          <button
                            type="button"
                            onClick={() => setShowWordSquares((v) => !v)}
                            className="text-[11px] font-medium text-indigo-600 dark:text-indigo-400 inline-flex items-center gap-1.5 hover:underline"
                          >
                            <Crop className="w-3 h-3" />
                            <span>
                              {showWordSquares ? 'Hide' : 'Show'} {detectedWords.length} Detected
                              Word Squares
                            </span>
                          </button>
                          <span className="text-[10px] text-zinc-400 font-mono-nums">
                            Fitted to 1:1 squares
                          </span>
                        </div>

                        {showWordSquares && (
                          <div className="flex items-center gap-2 overflow-x-auto p-2 bg-zinc-50 dark:bg-zinc-950/80 border border-zinc-200 dark:border-zinc-800 rounded-xl no-scrollbar">
                            {detectedWords.map((word, idx) => (
                              <div
                                key={word.id}
                                className="flex flex-col items-center gap-1 shrink-0 p-1 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg shadow-xs"
                              >
                                {word.thumbnailUrl ? (
                                  <img
                                    src={word.thumbnailUrl}
                                    alt={`Word ${idx + 1}`}
                                    className="w-12 h-12 rounded object-cover border border-zinc-100 dark:border-zinc-800"
                                  />
                                ) : (
                                  <div className="w-12 h-12 rounded bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center text-[10px] text-zinc-400">
                                    W{word.wordIndex + 1}
                                  </div>
                                )}
                                <span className="text-[10px] font-mono-nums text-zinc-500">
                                  W{word.wordIndex + 1}
                                </span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-zinc-600 dark:text-zinc-400 mb-1.5">
                      Optional reminder hint
                    </label>
                    <input
                      type="text"
                      value={promptText}
                      onChange={(e) => setPromptText(e.target.value)}
                      placeholder="e.g. Remind me tomorrow morning at 9am"
                      className="w-full min-h-[44px] px-3 bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-xl text-base sm:text-sm text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 focus:outline-none focus:border-indigo-500"
                    />
                  </div>

                  {sourceImage && (
                    <div className="grid grid-cols-2 gap-2 pt-1">
                      <button
                        type="button"
                        onClick={() => runExtractionPipeline(sourceImage)}
                        className="min-h-[44px] px-3 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 hover:bg-zinc-100 dark:hover:bg-zinc-800 text-xs font-medium text-zinc-700 dark:text-zinc-300 inline-flex items-center justify-center gap-1.5 whitespace-nowrap"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                        <span>Re-scan</span>
                      </button>
                      <button
                        type="button"
                        onClick={() =>
                          runExtractionPipeline(sourceImage, {
                            forceVision: true,
                          })
                        }
                        className="min-h-[44px] px-3 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 hover:bg-zinc-100 dark:hover:bg-zinc-800 text-xs font-medium text-zinc-700 dark:text-zinc-300 inline-flex items-center justify-center gap-1.5 whitespace-nowrap"
                      >
                        <Sparkles className="w-3.5 h-3.5" />
                        <span>Try Vision AI</span>
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Sticky Bottom Primary CTA */}
            {!isScanning && (
              <div className="p-4 border-t border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 pb-[max(1rem,env(safe-area-inset-bottom))] shrink-0">
                {draft ? (
                  <button
                    type="button"
                    disabled={isSaving}
                    onClick={handleSaveDraft}
                    className="w-full min-h-[48px] rounded-xl bg-indigo-600 hover:bg-indigo-700 active:scale-[0.99] disabled:opacity-50 text-white text-sm font-medium inline-flex items-center justify-center gap-2 transition-all"
                  >
                    {isSaving ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Saving Reminder...</span>
                      </>
                    ) : (
                      <span>Save Reminder</span>
                    )}
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={handleConvertTextToReminder}
                    className="w-full min-h-[48px] rounded-xl bg-indigo-600 hover:bg-indigo-700 active:scale-[0.99] text-white text-sm font-medium inline-flex items-center justify-center gap-2 transition-all"
                  >
                    <span>Convert to Reminder</span>
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Fullscreen Source / Processed Image Zoom Lightbox */}
      {lightboxOpen && sourceImage && (
        <div
          className="fixed inset-0 z-[60] flex flex-col items-center justify-center gap-3 p-4 bg-black/90 backdrop-blur-sm"
          onClick={() => setLightboxOpen(false)}
        >
          {processedImage && (
            <div
              className="inline-flex items-center gap-1 p-1 rounded-xl bg-zinc-900 border border-zinc-700"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                type="button"
                onClick={() => setPreviewMode('processed')}
                className={`min-h-[36px] px-3 rounded-lg text-xs font-medium transition-colors ${
                  previewMode === 'processed'
                    ? 'bg-white text-zinc-900'
                    : 'text-zinc-300 hover:text-white'
                }`}
              >
                Cleaned OCR Input
              </button>
              <button
                type="button"
                onClick={() => setPreviewMode('original')}
                className={`min-h-[36px] px-3 rounded-lg text-xs font-medium transition-colors ${
                  previewMode === 'original'
                    ? 'bg-white text-zinc-900'
                    : 'text-zinc-300 hover:text-white'
                }`}
              >
                Original Photo
              </button>
            </div>
          )}
          <button
            type="button"
            onClick={() => setLightboxOpen(false)}
            className="absolute top-4 right-4 min-h-[44px] min-w-[44px] flex items-center justify-center rounded-xl bg-zinc-900 text-white"
            aria-label="Close image preview"
          >
            <X className="w-5 h-5" />
          </button>
          <img
            src={previewMode === 'processed' && processedImage ? processedImage : sourceImage}
            alt="Full scanned note preview"
            referrerPolicy="no-referrer"
            className="max-w-full max-h-[82vh] object-contain rounded-xl bg-white"
          />
        </div>
      )}
    </div>
  );
};
