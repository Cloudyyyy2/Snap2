import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Camera,
  Upload,
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
} from 'lucide-react';
import { CameraCapture } from './CameraCapture';
import { Reminder, Category, Priority, OcrProfile, ChecklistItem } from '../types';

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
  detectedStyle: 'printed' | 'handwriting';
  engineLabel: string;
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

const OCR_PROFILE_STORAGE_KEY = 'snapremind_ocr_profile';
const RECENT_SCANS_KEY = 'snapremind_recent_scans';
const CATEGORIES: Category[] = ['Bills', 'Events', 'Tasks', 'Travel', 'Work', 'Personal', 'Health'];
const PRIORITIES: Priority[] = ['high', 'medium', 'low'];

function getStoredProfile(): OcrProfile {
  try {
    const val = localStorage.getItem(OCR_PROFILE_STORAGE_KEY);
    if (val === 'auto' || val === 'printed' || val === 'handwriting') {
      return val;
    }
  } catch {
    // ignore storage errors
  }
  return 'auto';
}

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
  const [ocrProfile, setOcrProfile] = useState<OcrProfile>(() => getStoredProfile());
  const [localModelReady, setLocalModelReady] = useState<boolean | null>(null);

  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [sourceImage, setSourceImage] = useState<string | null>(null);
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

  const [recentScans, setRecentScans] = useState<RecentScanItem[]>(() => getStoredRecentScans());

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const nativeCameraInputRef = useRef<HTMLInputElement | null>(null);

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
        }
      })
      .catch(() => {
        if (active) setLocalModelReady(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const handleProfileChange = (next: OcrProfile) => {
    setOcrProfile(next);
    try {
      localStorage.setItem(OCR_PROFILE_STORAGE_KEY, next);
    } catch {
      // ignore storage errors
    }
  };

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
    async (
      image: string,
      targetProfile: OcrProfile,
      options?: { forceHandwriting?: boolean; forceVision?: boolean }
    ) => {
      setSourceImage(image);
      setIsCameraOpen(false);
      setReviewSheetOpen(true);
      setIsScanning(true);
      setErrorMsg(null);
      setDraft(null);
      setScanMeta(null);

      try {
        const shouldTryLocal =
          !options?.forceVision &&
          localModelReady !== false &&
          (options?.forceHandwriting || targetProfile === 'handwriting' || targetProfile === 'auto');

        if (shouldTryLocal) {
          setStatusStep(
            options?.forceHandwriting || targetProfile === 'handwriting'
              ? 'Reading handwriting locally...'
              : 'Detecting note style locally...'
          );

          const res = await fetch('/api/local-ocr', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              image,
              ocrProfile: options?.forceHandwriting ? 'handwriting' : targetProfile,
              exportStages: true,
            }),
          });

          const localData = await res.json();
          if (!res.ok || !localData.success) {
            throw new Error(localData.error || `Local OCR failed (HTTP ${res.status})`);
          }

          if (localData.recognizer === 'trocr-host') {
            const linesCount = Array.isArray(localData.lines) ? localData.lines.length : 0;
            const label = 'Handwriting · TrOCR (Local)';
            setEditableText(localData.text || '');
            setScanMeta({
              linesCount,
              detectedStyle: localData.detectedStyle || 'handwriting',
              engineLabel: label,
            });
            recordRecentScan(image, localData.text || 'Handwritten note', label);
            if (!localData.text) {
              setErrorMsg(
                'No clear handwriting lines were recognized. Edit the note below or try Vision AI.'
              );
            }
            return;
          }
        }

        // Printed capture or Vision AI path: extract structured reminder directly
        setStatusStep('Extracting text & reminder details...');
        const res = await fetch('/api/extract-reminder', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image,
            prompt: promptText.trim() || undefined,
            ocrProfile: targetProfile,
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
          data.provider === 'local-parser'
            ? 'Offline · Local Parser'
            : `${targetProfile === 'handwriting' ? 'Handwriting' : 'Printed'} · Vision AI`;

        const extractedTextPreview =
          data.ocrText ||
          [data.data?.title, data.data?.description].filter(Boolean).join('\n');

        setEditableText(extractedTextPreview);
        setScanMeta({
          linesCount: extractedTextPreview ? extractedTextPreview.split('\n').length : 1,
          detectedStyle: targetProfile === 'handwriting' ? 'handwriting' : 'printed',
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
    [buildDraftFromApi, localModelReady, promptText, recordRecentScan]
  );

  // Pick up workspace drag-and-drop image passed from App
  useEffect(() => {
    if (externalImage) {
      runExtractionPipeline(externalImage, ocrProfile);
      onClearExternalImage?.();
    }
  }, [externalImage, ocrProfile, onClearExternalImage, runExtractionPipeline]);

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
              runExtractionPipeline(result, ocrProfile);
            }
          };
          reader.readAsDataURL(file);
          break;
        }
      }
    };
    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, [ocrProfile, runExtractionPipeline]);

  const readFileAndScan = (file: File) => {
    const reader = new FileReader();
    reader.onload = (ev) => {
      const result = ev.target?.result as string;
      if (result) {
        runExtractionPipeline(result, ocrProfile);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleConvertTextToReminder = async () => {
    if (!editableText.trim() && !promptText.trim()) {
      setErrorMsg('Enter or scan some text before converting to a reminder.');
      return;
    }

    setIsScanning(true);
    setStatusStep('Structuring reminder...');
    setErrorMsg(null);

    try {
      const res = await fetch('/api/extract-reminder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: sourceImage,
          ocrText: editableText.trim(),
          prompt: promptText.trim() || undefined,
          ocrProfile,
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
      const reminder: Reminder = {
        id: `rem-${Date.now()}`,
        title: draft.title.trim() || 'Scanned Reminder',
        description: draft.description.trim(),
        dueDate: draft.dueDate ? new Date(draft.dueDate).toISOString() : new Date().toISOString(),
        category: draft.category,
        priority: draft.priority,
        checklist: draft.checklist,
        tags: Array.from(
          new Set([
            ...draft.tags,
            scanMeta?.detectedStyle === 'handwriting' ? 'handwritten' : 'ocr',
          ])
        ),
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

  return (
    <div className="w-full max-w-4xl mx-auto px-4 pt-3 pb-24 lg:px-8 lg:py-8 space-y-5">
      {/* Studio Intro Strip */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <h1 className="font-display text-lg sm:text-xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
            Scan notes & screenshots
          </h1>
          <p className="text-xs sm:text-sm text-zinc-500 dark:text-zinc-400 mt-0.5">
            Capture handwriting or printed text and convert it into scheduled reminders.
          </p>
        </div>

        {sourceImage && !reviewSheetOpen && (
          <button
            type="button"
            onClick={() => setReviewSheetOpen(true)}
            className="min-h-[40px] px-3 rounded-xl bg-indigo-50 dark:bg-indigo-950/50 text-indigo-700 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800/60 text-xs font-medium whitespace-nowrap shrink-0"
          >
            Resume Scan
          </button>
        )}
      </div>

      {/* Primary Capture Surface (Camera Viewfinder or Dropzone) */}
      {isCameraOpen ? (
        <CameraCapture
          onCapture={(dataUrl) => runExtractionPipeline(dataUrl, ocrProfile)}
          onClose={() => setIsCameraOpen(false)}
        />
      ) : (
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const file = e.dataTransfer.files?.[0];
            if (file && file.type.startsWith('image/')) {
              readFileAndScan(file);
            }
          }}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              fileInputRef.current?.click();
            }
          }}
          className="border border-dashed border-zinc-300 dark:border-zinc-700 hover:border-indigo-500 dark:hover:border-indigo-500 rounded-2xl p-6 sm:p-10 text-center bg-zinc-50/80 dark:bg-zinc-900/50 transition-colors cursor-pointer flex flex-col items-center justify-center gap-3 min-h-[210px] sm:min-h-[250px]"
        >
          <div className="w-12 h-12 rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-indigo-600 dark:text-indigo-400 flex items-center justify-center">
            <Upload className="w-5 h-5" />
          </div>
          <div className="space-y-1 max-w-xs">
            <p className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
              Tap to choose a photo or paste screenshot
            </p>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 leading-relaxed">
              Works on cursive notebook pages, lined paper, bills, receipts, and event flyers.
            </p>
          </div>
        </div>
      )}

      {/* Hidden inputs: gallery picker + direct device camera capture */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) readFileAndScan(file);
          e.target.value = '';
        }}
        className="hidden"
      />
      <input
        ref={nativeCameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) readFileAndScan(file);
          e.target.value = '';
        }}
        className="hidden"
      />

      {/* Natural Thumb Zone Controls: Mode Switcher + Primary Camera & Photo CTAs */}
      {!isCameraOpen && (
        <div className="space-y-3">
          {/* 3-Way OCR Mode Control + Unboxed Engine Status */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
            <div className="grid grid-cols-3 gap-1 p-1 bg-zinc-100 dark:bg-zinc-900 border border-zinc-200/80 dark:border-zinc-800 rounded-xl">
              {(
                [
                  { id: 'auto', label: 'Auto' },
                  { id: 'printed', label: 'Printed' },
                  { id: 'handwriting', label: 'Handwriting' },
                ] as const
              ).map((mode) => {
                const active = ocrProfile === mode.id;
                return (
                  <button
                    key={mode.id}
                    type="button"
                    onClick={() => handleProfileChange(mode.id)}
                    className={`min-h-[40px] px-3 rounded-lg text-xs font-medium transition-colors whitespace-nowrap ${
                      active
                        ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-sm'
                        : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
                    }`}
                  >
                    {mode.label}
                  </button>
                );
              })}
            </div>

            <span className="text-xs text-zinc-500 dark:text-zinc-400 px-1">
              {localModelReady === true
                ? 'Auto-detects handwriting locally via TrOCR'
                : localModelReady === false
                  ? 'Vision AI & offline parser active'
                  : 'Checking local OCR engine...'}
            </span>
          </div>

          {/* 48px Thumb-Friendly Primary Capture Buttons */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            <button
              type="button"
              onClick={() => setIsCameraOpen(true)}
              className="min-h-[48px] px-4 rounded-xl bg-indigo-600 hover:bg-indigo-700 active:scale-[0.99] text-white text-sm font-medium inline-flex items-center justify-center gap-2 transition-all whitespace-nowrap"
            >
              <Camera className="w-4 h-4" />
              <span>Open Live Camera</span>
            </button>

            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => nativeCameraInputRef.current?.click()}
                className="min-h-[48px] px-3 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 hover:bg-zinc-50 dark:hover:bg-zinc-800/80 text-xs sm:text-sm font-medium text-zinc-800 dark:text-zinc-200 inline-flex items-center justify-center gap-1.5 transition-colors whitespace-nowrap"
              >
                <Camera className="w-4 h-4 text-zinc-500 dark:text-zinc-400" />
                <span>Device Cam</span>
              </button>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="min-h-[48px] px-3 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 hover:bg-zinc-50 dark:hover:bg-zinc-800/80 text-xs sm:text-sm font-medium text-zinc-800 dark:text-zinc-200 inline-flex items-center justify-center gap-1.5 transition-colors whitespace-nowrap"
              >
                <Upload className="w-4 h-4 text-zinc-500 dark:text-zinc-400" />
                <span>Gallery</span>
              </button>
            </div>
          </div>
        </div>
      )}

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
                onClick={() => runExtractionPipeline(item.image, ocrProfile)}
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

            {/* Sheet Header with Zoomable Source Thumbnail */}
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
                      src={sourceImage}
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
                    </div>
                    <textarea
                      rows={5}
                      value={editableText}
                      onChange={(e) => setEditableText(e.target.value)}
                      placeholder="Review or edit recognized text before converting..."
                      className="w-full p-3.5 bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-xl text-base sm:text-sm text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 focus:outline-none focus:border-indigo-500 leading-relaxed"
                    />
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
                        onClick={() =>
                          runExtractionPipeline(sourceImage, 'handwriting', {
                            forceHandwriting: true,
                          })
                        }
                        className="min-h-[44px] px-3 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 hover:bg-zinc-100 dark:hover:bg-zinc-800 text-xs font-medium text-zinc-700 dark:text-zinc-300 inline-flex items-center justify-center gap-1.5 whitespace-nowrap"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                        <span>Re-read handwriting</span>
                      </button>
                      <button
                        type="button"
                        onClick={() =>
                          runExtractionPipeline(sourceImage, ocrProfile, {
                            forceVision: true,
                          })
                        }
                        className="min-h-[44px] px-3 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 hover:bg-zinc-100 dark:hover:bg-zinc-800 text-xs font-medium text-zinc-700 dark:text-zinc-300 inline-flex items-center justify-center gap-1.5 whitespace-nowrap"
                      >
                        <Sparkles className="w-3.5 h-3.5" />
                        <span>Use Vision AI</span>
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

      {/* Fullscreen Source Image Zoom Lightbox */}
      {lightboxOpen && sourceImage && (
        <div
          className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/90 backdrop-blur-sm"
          onClick={() => setLightboxOpen(false)}
        >
          <button
            type="button"
            onClick={() => setLightboxOpen(false)}
            className="absolute top-4 right-4 min-h-[44px] min-w-[44px] flex items-center justify-center rounded-xl bg-zinc-900 text-white"
            aria-label="Close image preview"
          >
            <X className="w-5 h-5" />
          </button>
          <img
            src={sourceImage}
            alt="Full scanned note preview"
            referrerPolicy="no-referrer"
            className="max-w-full max-h-[88vh] object-contain rounded-xl"
          />
        </div>
      )}
    </div>
  );
};
