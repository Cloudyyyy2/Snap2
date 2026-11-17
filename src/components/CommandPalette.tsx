import React, { useState, useEffect, useRef } from 'react';
import {
  Search,
  Inbox,
  Calendar,
  Clock,
  Star,
  CheckCircle,
  Camera,
  Plus,
  Sun,
  Moon,
  ArrowRight,
  X,
} from 'lucide-react';
import { Reminder } from '../types';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  reminders: Reminder[];
  onSelectReminder: (reminder: Reminder) => void;
  onSelectView: (view: string) => void;
  onOpenScan: () => void;
  onOpenAdd: () => void;
  isDark: boolean;
  onToggleTheme: () => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  isOpen,
  onClose,
  reminders,
  onSelectReminder,
  onSelectView,
  onOpenScan,
  onOpenAdd,
  isDark,
  onToggleTheme,
}) => {
  const [query, setQuery] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setTimeout(() => inputRef.current?.focus(), 20);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const q = query.trim().toLowerCase();

  const quickActions = [
    {
      id: 'act-scan',
      label: 'Open OCR Studio',
      shortcut: 'S',
      icon: <Camera className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />,
      run: () => {
        onClose();
        onOpenScan();
      },
    },
    {
      id: 'act-add',
      label: 'Create New Reminder',
      shortcut: 'N',
      icon: <Plus className="w-4 h-4 text-zinc-600 dark:text-zinc-400" />,
      run: () => {
        onClose();
        onOpenAdd();
      },
    },
    {
      id: 'act-theme',
      label: isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode',
      shortcut: 'T',
      icon: isDark ? <Sun className="w-4 h-4 text-amber-500" /> : <Moon className="w-4 h-4 text-zinc-600" />,
      run: () => {
        onToggleTheme();
        onClose();
      },
    },
  ].filter((a) => !q || a.label.toLowerCase().includes(q));

  const viewActions = [
    { id: 'scan', label: 'Go to OCR Studio', icon: <Camera className="w-4 h-4" /> },
    { id: 'all', label: 'Go to Inbox', icon: <Inbox className="w-4 h-4" /> },
    { id: 'today', label: 'Go to Today', icon: <Calendar className="w-4 h-4" /> },
    { id: 'upcoming', label: 'Go to Upcoming', icon: <Clock className="w-4 h-4" /> },
    { id: 'starred', label: 'Go to Starred', icon: <Star className="w-4 h-4" /> },
    { id: 'completed', label: 'Go to Completed', icon: <CheckCircle className="w-4 h-4" /> },
  ].filter((v) => !q || v.label.toLowerCase().includes(q));

  const matchedReminders = reminders
    .filter((r) => {
      if (!q) return !r.completed;
      return (
        r.title.toLowerCase().includes(q) ||
        r.description.toLowerCase().includes(q) ||
        r.category.toLowerCase().includes(q) ||
        r.tags.some((t) => t.toLowerCase().includes(q))
      );
    })
    .slice(0, 6);

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[12vh] px-4 bg-zinc-950/50 dark:bg-black/70 backdrop-blur-[2px]"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 px-4 py-3 border-b border-zinc-200 dark:border-zinc-800">
          <Search className="w-4 h-4 text-zinc-400 shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search reminders, jump to views, or run actions..."
            className="flex-1 bg-transparent text-sm text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 dark:placeholder-zinc-500 focus:outline-none"
          />
          <button
            onClick={onClose}
            className="p-1 text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 rounded-md"
            aria-label="Close command menu"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="max-h-[60vh] overflow-y-auto p-2 space-y-3">
          {quickActions.length > 0 && (
            <div>
              <div className="px-2.5 py-1 text-[11px] font-medium text-zinc-400 dark:text-zinc-500">
                Actions
              </div>
              {quickActions.map((act) => (
                <button
                  key={act.id}
                  onClick={act.run}
                  className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-sm text-zinc-700 dark:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-zinc-800/80 transition-colors"
                >
                  <span className="flex items-center gap-2.5">
                    {act.icon}
                    <span>{act.label}</span>
                  </span>
                  <kbd className="font-mono-nums text-[11px] text-zinc-400 dark:text-zinc-500 px-1.5 py-0.5 border border-zinc-200 dark:border-zinc-700 rounded">
                    {act.shortcut}
                  </kbd>
                </button>
              ))}
            </div>
          )}

          {matchedReminders.length > 0 && (
            <div>
              <div className="px-2.5 py-1 text-[11px] font-medium text-zinc-400 dark:text-zinc-500">
                Reminders
              </div>
              {matchedReminders.map((r) => (
                <button
                  key={r.id}
                  onClick={() => {
                    onSelectReminder(r);
                    onClose();
                  }}
                  className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-left hover:bg-zinc-100 dark:hover:bg-zinc-800/80 transition-colors group"
                >
                  <div className="min-w-0 pr-3">
                    <p className="text-sm font-medium text-zinc-900 dark:text-zinc-100 truncate">
                      {r.title}
                    </p>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400 truncate">
                      {r.category} ·{' '}
                      <span className="tabular-nums">
                        {new Date(r.dueDate).toLocaleDateString(undefined, {
                          month: 'short',
                          day: 'numeric',
                        })}
                      </span>
                    </p>
                  </div>
                  <ArrowRight className="w-3.5 h-3.5 text-zinc-400 opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
                </button>
              ))}
            </div>
          )}

          {viewActions.length > 0 && (
            <div>
              <div className="px-2.5 py-1 text-[11px] font-medium text-zinc-400 dark:text-zinc-500">
                Views
              </div>
              {viewActions.map((v) => (
                <button
                  key={v.id}
                  onClick={() => {
                    onSelectView(v.id);
                    onClose();
                  }}
                  className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800/80 transition-colors"
                >
                  <span className="text-zinc-400">{v.icon}</span>
                  <span>{v.label}</span>
                </button>
              ))}
            </div>
          )}

          {quickActions.length === 0 && matchedReminders.length === 0 && viewActions.length === 0 && (
            <div className="py-8 text-center text-xs text-zinc-500 dark:text-zinc-400">
              No matching commands or reminders.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
