import React, { useEffect, useState } from 'react';
import { Layers, Loader2 } from 'lucide-react';

interface OcrExportRun {
  name: string;
  files: string[];
  urls: string[];
}

interface OcrXRayViewerProps {
  exportDirName?: string | null;
}

function formatStageLabel(filename: string): string {
  if (filename.startsWith('01_')) return '01 Original';
  if (filename.startsWith('02_')) return '02 Deskewed';
  if (filename.startsWith('03b_')) return '03b Flattened';
  if (filename.startsWith('03c_')) return '03c Lines Removed';
  if (filename.startsWith('04_')) return '04 Contrast';
  if (filename.startsWith('05_')) return '05 Binary';
  if (filename.startsWith('06_line_')) {
    return filename.replace('.png', '').replace('06_line_', 'Line ');
  }
  if (filename.startsWith('07_')) return '07 OCR Overlay';
  return filename.replace('.png', '');
}

export const OcrXRayViewer: React.FC<OcrXRayViewerProps> = ({ exportDirName }) => {
  const [stages, setStages] = useState<Array<{ file: string; url: string; label: string }>>([]);
  const [activeIdx, setActiveIdx] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    fetch('/api/ocr-exports')
      .then((r) => r.json())
      .then((data) => {
        if (!active) return;
        const runs: OcrExportRun[] = Array.isArray(data?.runs) ? data.runs : [];
        const match = exportDirName
          ? runs.find((r) => r.name === exportDirName) || runs[0]
          : runs[0];
        if (!match) {
          setStages([]);
          setError('No exported OCR stage images found yet.');
        } else {
          const pngStages = match.files
            .map((file, i) => ({ file, url: match.urls[i], label: formatStageLabel(file) }))
            .filter((s) => s.file.endsWith('.png'));
          setStages(pngStages);
          setActiveIdx(0);
        }
      })
      .catch(() => {
        if (active) setError('Could not load OCR stage images.');
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [exportDirName]);

  if (loading) {
    return (
      <div className="p-3 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 flex items-center gap-2 text-xs text-zinc-500">
        <Loader2 className="w-3.5 h-3.5 animate-spin" />
        <span>Loading OCR X-Ray stages...</span>
      </div>
    );
  }

  if (error || stages.length === 0) {
    return (
      <div className="p-3 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 text-xs text-zinc-500">
        {error || 'No stage images available.'}
      </div>
    );
  }

  const current = stages[Math.min(activeIdx, stages.length - 1)];

  return (
    <div className="rounded-xl border border-indigo-200/80 dark:border-indigo-900/60 bg-indigo-50/40 dark:bg-indigo-950/20 p-3 space-y-2.5">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-indigo-700 dark:text-indigo-300">
          <Layers className="w-3.5 h-3.5" />
          <span>OCR X-Ray Stage Scrubber</span>
        </div>
        <span className="text-[11px] font-mono text-indigo-600 dark:text-indigo-400">
          {current.label} ({activeIdx + 1}/{stages.length})
        </span>
      </div>

      <div className="rounded-lg overflow-hidden border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 h-44 flex items-center justify-center">
        <img
          src={current.url}
          alt={`OCR stage ${current.label}`}
          className="h-full w-auto max-w-full object-contain"
        />
      </div>

      <input
        type="range"
        min={0}
        max={stages.length - 1}
        value={activeIdx}
        onChange={(e) => setActiveIdx(Number(e.target.value))}
        aria-label="Scrub OCR processing stages"
        className="w-full accent-indigo-600 cursor-pointer"
      />

      <div className="flex flex-wrap gap-1">
        {stages.map((stage, idx) => (
          <button
            key={stage.file}
            type="button"
            onClick={() => setActiveIdx(idx)}
            className={`px-2 py-0.5 rounded text-[10px] font-medium border transition-colors ${
              idx === activeIdx
                ? 'bg-indigo-600 text-white border-indigo-600'
                : 'bg-white dark:bg-zinc-900 text-zinc-600 dark:text-zinc-300 border-zinc-200 dark:border-zinc-700 hover:border-indigo-400'
            }`}
          >
            {stage.label}
          </button>
        ))}
      </div>
    </div>
  );
};
