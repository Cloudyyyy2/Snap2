import React, { useState } from 'react';
import { X, Upload, Camera, Loader2 } from 'lucide-react';
import { Reminder } from '../types';

interface UploadModalProps {
  onClose: () => void;
  onReminderExtracted: (reminder: Reminder) => void;
  initialImage?: string | null;
}

export const UploadModal: React.FC<UploadModalProps> = ({
  onClose,
  onReminderExtracted,
  initialImage = null,
}) => {
  const [imagePreview, setImagePreview] = useState<string | null>(initialImage);
  const [imageBase64, setImageBase64] = useState<string | null>(initialImage);
  const [promptText, setPromptText] = useState('');
  const [isExtracting, setIsExtracting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const readFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (uploadEvent) => {
      const result = uploadEvent.target?.result as string;
      setImagePreview(result);
      setImageBase64(result);
      setErrorMsg(null);
    };
    reader.readAsDataURL(file);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) readFile(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) readFile(file);
  };

  const handleExtractWithAI = async () => {
    if (!imageBase64 && !promptText.trim()) {
      setErrorMsg('Upload a screenshot or enter details to extract.');
      return;
    }

    setIsExtracting(true);
    setErrorMsg(null);

    try {
      const res = await fetch('/api/extract-reminder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: imageBase64,
          prompt: promptText.trim() || 'Extract reminder details from this image.',
        }),
      });

      const contentType = res.headers.get('content-type') || '';
      const data = contentType.includes('application/json')
        ? await res.json()
        : { success: false, error: await res.text() };

      if (!res.ok || !data.success) {
        throw new Error(data.error || `Failed to extract reminder (HTTP ${res.status})`);
      }

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
        tags: Array.isArray(parsed.tags) ? parsed.tags : ['screenshot'],
        image: imageBase64 || undefined,
        completed: false,
        starred: false,
        createdAt: new Date().toISOString(),
      };

      onReminderExtracted(newReminder);
      onClose();
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || 'AI extraction failed. Please try again.');
    } finally {
      setIsExtracting(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-zinc-950/50 dark:bg-black/70 backdrop-blur-[2px]"
      onClick={onClose}
    >
      <div
        className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl w-full max-w-lg overflow-hidden shadow-2xl flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-200 dark:border-zinc-800">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-50 dark:bg-indigo-950/50 text-indigo-600 dark:text-indigo-400 border border-indigo-200 dark:border-indigo-800/60 flex items-center justify-center">
              <Camera className="w-4 h-4" />
            </div>
            <div>
              <h2 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                Scan Screenshot
              </h2>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Extract due dates, action steps, and notes from any image
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

        <div className="p-5 space-y-4">
          {errorMsg && (
            <div className="p-3 bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800/60 rounded-lg text-xs text-rose-700 dark:text-rose-300">
              {errorMsg}
            </div>
          )}

          {imagePreview ? (
            <div className="relative rounded-lg overflow-hidden border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 h-52 flex items-center justify-center">
              <img
                src={imagePreview}
                alt="Uploaded screenshot preview"
                referrerPolicy="no-referrer"
                className="h-full object-contain"
              />
              <button
                type="button"
                onClick={() => {
                  setImagePreview(null);
                  setImageBase64(null);
                }}
                className="absolute top-2.5 right-2.5 bg-white/95 dark:bg-zinc-900/95 text-zinc-700 dark:text-zinc-200 px-2.5 py-1 rounded-md border border-zinc-200 dark:border-zinc-700 hover:text-rose-600 text-xs font-medium inline-flex items-center gap-1 shadow-sm"
              >
                <X className="w-3.5 h-3.5" /> Remove
              </button>
            </div>
          ) : (
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              className="border border-dashed border-zinc-300 dark:border-zinc-700 hover:border-indigo-500 dark:hover:border-indigo-500 rounded-xl p-8 text-center bg-zinc-50/70 dark:bg-zinc-950/50 transition-colors cursor-pointer flex flex-col items-center justify-center gap-2.5 relative"
            >
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="absolute inset-0 opacity-0 cursor-pointer"
              />
              <div className="w-10 h-10 rounded-lg bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-zinc-600 dark:text-zinc-300 flex items-center justify-center">
                <Upload className="w-5 h-5" />
              </div>
              <div>
                <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200">
                  Drop a screenshot here or <span className="text-indigo-600 dark:text-indigo-400">browse</span>
                </p>
                <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">
                  Bills, receipts, flyers, tickets, or notes (PNG, JPG, WEBP)
                </p>
              </div>
            </div>
          )}

          <div>
            <label className="block text-xs font-medium text-zinc-600 dark:text-zinc-400 mb-1.5">
              Optional context
            </label>
            <input
              type="text"
              value={promptText}
              onChange={(e) => setPromptText(e.target.value)}
              placeholder="e.g. Remind me 2 days before the due date"
              className="w-full h-9 px-3 bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-lg text-xs text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 focus:outline-none focus:border-indigo-500"
            />
          </div>
        </div>

        <div className="px-5 py-3.5 border-t border-zinc-200 dark:border-zinc-800 bg-zinc-50/60 dark:bg-zinc-950/40 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="px-3.5 py-2 text-xs font-medium text-zinc-600 dark:text-zinc-300 hover:bg-zinc-200/60 dark:hover:bg-zinc-800 rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleExtractWithAI}
            disabled={isExtracting}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-xs font-medium rounded-lg transition-colors inline-flex items-center gap-2 whitespace-nowrap"
          >
            {isExtracting ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Extracting...</span>
              </>
            ) : (
              <span>Extract Reminder</span>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
