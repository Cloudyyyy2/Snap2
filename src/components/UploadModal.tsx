import React, { useState, useEffect } from 'react';
import { X, Camera, Loader2, RotateCcw, Sparkles, ShieldCheck, HelpCircle, FolderOpen, Layers } from 'lucide-react';
import { Reminder, DetectedStyle, OcrLineResult, CaptureMode } from '../types';
import { CameraCapture } from './CameraCapture';
import { OcrXRayViewer } from './OcrXRayViewer';

interface UploadModalProps {
  onClose: () => void;
  onReminderExtracted: (reminder: Reminder) => void;
  initialImage?: string | null;
}

interface LocalOcrState {
  linesCount: number;
  detectedStyle: DetectedStyle;
  engineLabel: string;
}

export const UploadModal: React.FC<UploadModalProps> = ({
  onClose,
  onReminderExtracted,
  initialImage = null,
}) => {
  const [imagePreview, setImagePreview] = useState<string | null>(initialImage);
  const [imageBase64, setImageBase64] = useState<string | null>(initialImage);
  const [promptText, setPromptText] = useState('');
  const [captureMode, setCaptureMode] = useState<CaptureMode>('auto');
  const [showXRay, setShowXRay] = useState(false);
  const [localModelReady, setLocalModelReady] = useState<boolean | null>(null);
  const [offlineOnly, setOfflineOnly] = useState(false);
  const [visionConfigured, setVisionConfigured] = useState(false);

  const [ocrState, setOcrState] = useState<LocalOcrState | null>(null);
  const [ocrLines, setOcrLines] = useState<OcrLineResult[]>([]);
  const [highlightedLineIdx, setHighlightedLineIdx] = useState<number | null>(null);
  const [exportDirName, setExportDirName] = useState<string | null>(null);
  const [needsUserPurpose, setNeedsUserPurpose] = useState(false);
  const [editableText, setEditableText] = useState('');
  const [isExtracting, setIsExtracting] = useState(false);
  const [statusStep, setStatusStep] = useState<string>('');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetch('/api/local-ocr/status')
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (active && data) {
          setLocalModelReady(
            data.handwritingModel === 'ready' || data.printedModel === 'ready'
          );
          setOfflineOnly(Boolean(data.offlineOnly));
          setVisionConfigured(Boolean(data.visionConfigured));
        }
      })
      .catch(() => {
        if (active) setLocalModelReady(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const readFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (uploadEvent) => {
      const result = uploadEvent.target?.result as string;
      setImagePreview(result);
      setImageBase64(result);
      setOcrState(null);
      setOcrLines([]);
      setHighlightedLineIdx(null);
      setExportDirName(null);
      setNeedsUserPurpose(false);
      setEditableText('');
      setErrorMsg(null);
    };
    reader.readAsDataURL(file);
  };

  const finalizeReminderFromApi = (data: any, styleTag?: string) => {
    const parsed = data.data || {};
    const validCategories = ['Bills', 'Events', 'Tasks', 'Travel', 'Work', 'Personal', 'Health'];
    const validPriorities = ['high', 'medium', 'low'];
    const category = validCategories.includes(parsed.category) ? parsed.category : 'Tasks';
    const priority = validPriorities.includes(parsed.priority) ? parsed.priority : 'medium';

    const parsedDate = parsed.dueDate ? new Date(parsed.dueDate) : null;
    const dueDate =
      parsedDate && !isNaN(parsedDate.getTime())
        ? parsedDate.toISOString()
        : new Date(Date.now() + 86400000).toISOString();

    const baseTags = Array.isArray(parsed.tags) ? parsed.tags : ['screenshot'];
    const tags = styleTag && !baseTags.includes(styleTag) ? [...baseTags, styleTag] : baseTags;

    const newReminder: Reminder = {
      id: `rem-${Date.now()}`,
      title: parsed.title || 'Untitled Reminder',
      description: parsed.description || '',
      dueDate,
      category,
      priority,
      checklist: Array.isArray(parsed.checklist)
        ? parsed.checklist
            .map((c: any, idx: number) => ({
              id: `chk-${Date.now()}-${idx}`,
              text: typeof c === 'string' ? c : c?.text || '',
              completed: Boolean(c?.completed),
            }))
            .filter((c: { text: string }) => c.text)
        : [],
      tags,
      image: imageBase64 || undefined,
      exportDirName: exportDirName || undefined,
      ocrText: editableText.trim() || undefined,
      ocrEngine: ocrState?.engineLabel,
      completed: false,
      starred: false,
      createdAt: new Date().toISOString(),
    };

    onReminderExtracted(newReminder);
    onClose();
  };

  const runAutoLocalOcr = async (): Promise<boolean> => {
    if (!imageBase64) return false;

    setStatusStep('Auto-detecting printed & handwritten text offline...');
    const res = await fetch('/api/local-ocr', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: imageBase64,
        ocrProfile: 'auto',
        captureMode,
        exportStages: true,
      }),
    });

    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.error || `Local OCR failed (HTTP ${res.status})`);
    }

    if (data.recognizer !== 'vision-fallback') {
      const lines: OcrLineResult[] = Array.isArray(data.lines) ? data.lines : [];
      const linesCount = lines.length;
      const engineLabel =
        data.recognizer === 'python-local-ocr'
          ? 'Python Local OCR · Offline'
          : data.recognizer === 'hybrid-host'
            ? 'Auto · Mixed (Offline)'
            : data.recognizer === 'tesseract-host'
              ? 'Auto · Printed (Offline)'
              : 'Auto · Handwriting (Offline)';

      setEditableText(data.text || '');
      setOcrLines(lines);
      setHighlightedLineIdx(null);
      setExportDirName(data.exportDirName || null);
      const isObjectOrBlank = Boolean(data.needsUserPurpose || !data.text?.trim());
      setNeedsUserPurpose(isObjectOrBlank);
      setOcrState({
        linesCount,
        detectedStyle: data.detectedStyle || 'handwriting',
        engineLabel,
      });
      return true;
    }

    return false;
  };

  const extractViaServer = async (overrideOcrText?: string, forceVisionImage = false) => {
    setStatusStep('Structuring reminder...');
    const res = await fetch('/api/extract-reminder', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: forceVisionImage ? imageBase64 : overrideOcrText !== undefined ? undefined : imageBase64,
        prompt: promptText.trim() || undefined,
        ocrProfile: 'auto',
        captureMode,
        ocrText: overrideOcrText,
      }),
    });

    const contentType = res.headers.get('content-type') || '';
    const data = contentType.includes('application/json')
      ? await res.json()
      : { success: false, error: await res.text() };

    if (!res.ok || !data.success) {
      throw new Error(data.error || `Failed to extract reminder (HTTP ${res.status})`);
    }

    if (data.needsUserPurpose && !data.data) {
      setNeedsUserPurpose(true);
      setExportDirName(data.exportDirName || null);
      setOcrState({
        linesCount: 0,
        detectedStyle: 'printed',
        engineLabel: 'Object / No-Text Photo (Offline)',
      });
      return;
    }

    const tag =
      ocrState?.detectedStyle === 'handwriting' || ocrState?.detectedStyle === 'mixed'
        ? 'handwritten'
        : needsUserPurpose
          ? 'photo-note'
          : undefined;
    finalizeReminderFromApi(data, tag);
  };

  const handlePrimaryAction = async () => {
    if (!imageBase64 && !promptText.trim() && !editableText.trim()) {
      setErrorMsg('Capture or upload a note or enter details to extract.');
      return;
    }

    setIsExtracting(true);
    setErrorMsg(null);

    try {
      if (ocrState !== null) {
        if (!editableText.trim() && !promptText.trim()) {
          setErrorMsg('Please enter or edit the note text before creating a reminder.');
          setIsExtracting(false);
          setStatusStep('');
          return;
        }
        await extractViaServer(editableText.trim());
        return;
      }

      if (imageBase64 && localModelReady !== false) {
        const handledByLocal = await runAutoLocalOcr();
        if (handledByLocal) {
          return;
        }
      }

      await extractViaServer(undefined, true);
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || 'Extraction failed. Please try again.');
    } finally {
      setIsExtracting(false);
      setStatusStep('');
    }
  };

  const handleRescanAuto = async () => {
    if (!imageBase64) return;
    setIsExtracting(true);
    setErrorMsg(null);
    try {
      const handled = await runAutoLocalOcr();
      if (!handled) {
        setErrorMsg('Offline OCR models are unavailable; using Vision AI instead.');
      }
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || 'Could not re-scan image.');
    } finally {
      setIsExtracting(false);
      setStatusStep('');
    }
  };

  const handleFallbackVisionAi = async () => {
    if (!imageBase64) return;
    setIsExtracting(true);
    setErrorMsg(null);
    try {
      await extractViaServer(undefined, true);
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || 'Vision AI extraction failed.');
    } finally {
      setIsExtracting(false);
      setStatusStep('');
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-zinc-950/50 dark:bg-black/70 backdrop-blur-[2px]"
      onClick={onClose}
    >
      <div
        className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl w-full max-w-lg overflow-hidden shadow-2xl flex flex-col max-h-[92vh]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-200 dark:border-zinc-800 shrink-0">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-50 dark:bg-indigo-950/50 text-indigo-600 dark:text-indigo-400 border border-indigo-200 dark:border-indigo-800/60 flex items-center justify-center">
              <Camera className="w-4 h-4" />
            </div>
            <div>
              <h2 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                Scan Note or Screenshot
              </h2>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Auto-detects printed, handwritten, and mixed pages offline
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 rounded-lg"
            aria-label="Close modal"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 space-y-4 overflow-y-auto">
          {errorMsg && (
            <div className="p-3 bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800/60 rounded-lg text-xs text-rose-700 dark:text-rose-300">
              {errorMsg}
            </div>
          )}

          <div className="flex flex-wrap items-center gap-1.5">
            {(
              [
                { id: 'auto', label: 'Auto' },
                { id: 'grocery', label: 'Grocery' },
                { id: 'receipt', label: 'Receipt' },
                { id: 'recipe', label: 'Recipe' },
                { id: 'fridge', label: 'Fridge' },
              ] as const
            ).map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => setCaptureMode(m.id)}
                className={`px-2.5 py-1 rounded-full text-[11px] font-medium border transition-colors ${
                  captureMode === m.id
                    ? 'bg-indigo-600 text-white border-indigo-600'
                    : 'bg-zinc-50 dark:bg-zinc-950 text-zinc-600 dark:text-zinc-300 border-zinc-200 dark:border-zinc-800 hover:border-indigo-400'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>

          {imagePreview ? (
            <div className="relative rounded-lg overflow-hidden border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 h-52 flex items-center justify-center">
              <div className="relative inline-block h-full max-w-full">
                <img
                  src={imagePreview}
                  alt="Uploaded note or screenshot preview"
                  referrerPolicy="no-referrer"
                  className="h-full w-auto max-w-full object-contain block"
                />
                {ocrLines.map((line, idx) => {
                  if (!line.normBox) return null;
                  const isActive = highlightedLineIdx === idx;
                  return (
                    <div
                      key={idx}
                      title={line.text}
                      onMouseEnter={() => setHighlightedLineIdx(idx)}
                      onMouseLeave={() => setHighlightedLineIdx(null)}
                      onClick={() => setHighlightedLineIdx(isActive ? null : idx)}
                      style={{
                        left: `${line.normBox.x * 100}%`,
                        top: `${line.normBox.y * 100}%`,
                        width: `${line.normBox.width * 100}%`,
                        height: `${line.normBox.height * 100}%`,
                      }}
                      className={`absolute border transition-colors cursor-pointer rounded-[2px] ${
                        isActive
                          ? 'border-amber-400 bg-amber-400/25 ring-2 ring-amber-400/60'
                          : 'border-indigo-500/70 bg-indigo-500/10 hover:bg-indigo-500/20'
                      }`}
                    />
                  );
                })}
              </div>
              {offlineOnly && (
                <span className="absolute top-2.5 left-2.5 bg-emerald-600/90 text-white px-2 py-0.5 rounded-md text-[10px] font-medium inline-flex items-center gap-1 shadow-sm">
                  <ShieldCheck className="w-3 h-3" /> Offline In-House
                </span>
              )}
              <button
                type="button"
                onClick={() => {
                  setImagePreview(null);
                  setImageBase64(null);
                  setOcrState(null);
                  setOcrLines([]);
                  setHighlightedLineIdx(null);
                  setExportDirName(null);
                  setNeedsUserPurpose(false);
                  setEditableText('');
                  setErrorMsg(null);
                }}
                className="absolute top-2.5 right-2.5 bg-white/95 dark:bg-zinc-900/95 text-zinc-700 dark:text-zinc-200 px-2.5 py-1 rounded-md border border-zinc-200 dark:border-zinc-700 hover:text-rose-600 text-xs font-medium inline-flex items-center gap-1 shadow-sm"
              >
                <X className="w-3.5 h-3.5" /> Retake
              </button>
            </div>
          ) : (
            <CameraCapture
              embedded
              paused={isExtracting}
              onCapture={(dataUrl) => {
                setImagePreview(dataUrl);
                setImageBase64(dataUrl);
                setOcrState(null);
                setEditableText('');
                setErrorMsg(null);
              }}
              onUploadFile={readFile}
              statusNote={
                localModelReady === true
                  ? 'Offline auto-OCR ready'
                  : localModelReady === false
                    ? 'Vision AI fallback'
                    : 'Checking offline OCR...'
              }
            />
          )}

          {needsUserPurpose && (
            <div className="p-3 bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800/60 rounded-lg text-xs text-amber-800 dark:text-amber-200 flex items-start gap-2">
              <HelpCircle className="w-4 h-4 shrink-0 mt-0.5 text-amber-600 dark:text-amber-400" />
              <div>
                <p className="font-semibold">No readable text found — what is this photo for?</p>
                <p className="text-[11px] opacity-85 mt-0.5">
                  Describe what you want to remember below (e.g. &ldquo;Buy replacement water filter today at 5pm&rdquo;) and tap Create Reminder.
                </p>
              </div>
            </div>
          )}

          {/* OCR-to-Image Line Reflection Chips */}
          {ocrLines.length > 0 && (
            <div className="space-y-1">
              <div className="flex items-center justify-between text-[11px] text-zinc-500 dark:text-zinc-400 gap-2">
                <span>Tap or hover a line to highlight it on the photo</span>
                {exportDirName && (
                  <button
                    type="button"
                    onClick={() => setShowXRay((v) => !v)}
                    className="inline-flex items-center gap-1 font-medium text-indigo-600 dark:text-indigo-400 hover:underline shrink-0"
                  >
                    <Layers className="w-3 h-3" />
                    <span>{showXRay ? 'Hide X-Ray' : 'X-Ray Steps'}</span>
                  </button>
                )}
              </div>
              <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto">
                {ocrLines.map((line, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onMouseEnter={() => setHighlightedLineIdx(idx)}
                    onMouseLeave={() => setHighlightedLineIdx(null)}
                    onClick={() => setHighlightedLineIdx(highlightedLineIdx === idx ? null : idx)}
                    className={`px-2 py-0.5 rounded text-[11px] border transition-colors text-left ${
                      highlightedLineIdx === idx
                        ? 'bg-amber-100 dark:bg-amber-900/50 border-amber-400 text-amber-900 dark:text-amber-100'
                        : 'bg-zinc-100 dark:bg-zinc-800 border-zinc-200 dark:border-zinc-700 text-zinc-700 dark:text-zinc-300'
                    }`}
                  >
                    {line.text}
                  </button>
                ))}
              </div>
            </div>
          )}

          {showXRay && exportDirName && <OcrXRayViewer exportDirName={exportDirName} />}

          {/* Editable OCR Text Review (shown after local OCR pass) */}
          {ocrState && !needsUserPurpose && (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <label className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                  Extracted text (editable)
                </label>
                <span className="text-[11px] text-zinc-500 dark:text-zinc-400 font-mono-nums">
                  {ocrState.linesCount} {ocrState.linesCount === 1 ? 'line' : 'lines'} ·{' '}
                  {ocrState.engineLabel}
                </span>
              </div>
              <textarea
                rows={4}
                value={editableText}
                onChange={(e) => setEditableText(e.target.value)}
                placeholder="Review or edit recognized text..."
                className="w-full p-3 bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-lg text-xs text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 focus:outline-none focus:border-indigo-500 leading-relaxed"
              />
              <div className="flex items-center justify-between gap-2 pt-0.5">
                <button
                  type="button"
                  disabled={isExtracting}
                  onClick={handleRescanAuto}
                  className="text-[11px] font-medium text-zinc-600 dark:text-zinc-400 hover:text-indigo-600 dark:hover:text-indigo-400 inline-flex items-center gap-1 whitespace-nowrap"
                >
                  <RotateCcw className="w-3 h-3" />
                  <span>Re-scan offline</span>
                </button>
                {!offlineOnly && visionConfigured && (
                  <button
                    type="button"
                    disabled={isExtracting}
                    onClick={handleFallbackVisionAi}
                    className="text-[11px] font-medium text-zinc-600 dark:text-zinc-400 hover:text-indigo-600 dark:hover:text-indigo-400 inline-flex items-center gap-1 whitespace-nowrap"
                  >
                    <Sparkles className="w-3 h-3" />
                    <span>Use Vision AI instead</span>
                  </button>
                )}
              </div>
            </div>
          )}

          <div>
            <label className="block text-xs font-medium text-zinc-600 dark:text-zinc-400 mb-1.5">
              {needsUserPurpose ? 'What is this photo for? (Required)' : 'Optional context'}
            </label>
            <input
              type="text"
              value={promptText}
              onChange={(e) => setPromptText(e.target.value)}
              placeholder={
                needsUserPurpose
                  ? 'e.g. Buy replacement water filter today at 5pm'
                  : 'e.g. Remind me 2 days before the due date'
              }
              className={`w-full h-9 px-3 bg-zinc-50 dark:bg-zinc-950 border rounded-lg text-xs text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 focus:outline-none focus:border-indigo-500 ${
                needsUserPurpose
                  ? 'border-amber-400 dark:border-amber-600'
                  : 'border-zinc-200 dark:border-zinc-800'
              }`}
            />
            {(needsUserPurpose || captureMode === 'fridge') && (
              <div className="flex flex-wrap gap-1.5 mt-2">
                {['Use in 2 days', 'Use this week', 'Buy replacement today'].map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    onClick={() => setPromptText(preset)}
                    className="px-2 py-0.5 rounded-md text-[11px] bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300 hover:bg-indigo-100 dark:hover:bg-indigo-900/40"
                  >
                    {preset}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="px-5 py-3.5 border-t border-zinc-200 dark:border-zinc-800 bg-zinc-50/60 dark:bg-zinc-950/40 flex items-center justify-between gap-2 shrink-0">
          <div>
            {imageBase64 && !ocrState && localModelReady && (
              <button
                type="button"
                disabled={isExtracting}
                onClick={handleRescanAuto}
                className="px-2.5 py-1.5 text-xs font-medium text-zinc-600 dark:text-zinc-300 hover:text-indigo-600 dark:hover:text-indigo-400 hover:bg-zinc-200/50 dark:hover:bg-zinc-800 rounded-lg transition-colors inline-flex items-center gap-1.5 whitespace-nowrap shrink-0"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Re-scan</span>
              </button>
            )}
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-3.5 py-2 text-xs font-medium text-zinc-600 dark:text-zinc-300 hover:bg-zinc-200/60 dark:hover:bg-zinc-800 rounded-lg transition-colors whitespace-nowrap shrink-0"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handlePrimaryAction}
              disabled={isExtracting}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-xs font-medium rounded-lg transition-colors inline-flex items-center gap-2 whitespace-nowrap shrink-0"
            >
              {isExtracting ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>{statusStep || 'Extracting...'}</span>
                </>
              ) : ocrState ? (
                <span>Create Reminder</span>
              ) : (
                <span>Extract Reminder</span>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
