import React, { useState, useEffect } from 'react';
import {
  X,
  Trash2,
  CheckCircle2,
  Circle,
  Star,
  Plus,
  CheckSquare,
  Square,
  Send,
  Maximize2,
} from 'lucide-react';
import { Reminder, Category, Priority, ChecklistItem } from '../types';

interface ReminderInspectorProps {
  reminder: Reminder;
  onClose: () => void;
  onUpdate: (updated: Reminder) => void;
  onDelete: (id: string) => void;
  onSendPushReminder?: (reminder: Reminder) => void;
  pushEnabled: boolean;
}

const CATEGORIES: Category[] = ['Bills', 'Events', 'Tasks', 'Travel', 'Work', 'Personal', 'Health'];
const PRIORITIES: Priority[] = ['high', 'medium', 'low'];

export const ReminderInspector: React.FC<ReminderInspectorProps> = ({
  reminder,
  onClose,
  onUpdate,
  onDelete,
  onSendPushReminder,
  pushEnabled,
}) => {
  const [title, setTitle] = useState(reminder.title);
  const [description, setDescription] = useState(reminder.description);
  const [dueDate, setDueDate] = useState(reminder.dueDate ? reminder.dueDate.slice(0, 16) : '');
  const [newChecklistText, setNewChecklistText] = useState('');
  const [newTagText, setNewTagText] = useState('');
  const [lightboxOpen, setLightboxOpen] = useState(false);

  useEffect(() => {
    setTitle(reminder.title);
    setDescription(reminder.description);
    setDueDate(reminder.dueDate ? reminder.dueDate.slice(0, 16) : '');
    setNewChecklistText('');
    setNewTagText('');
    setLightboxOpen(false);
  }, [reminder.id, reminder.title, reminder.description, reminder.dueDate]);

  const commitTextFields = () => {
    const trimmedTitle = title.trim() || reminder.title;
    if (trimmedTitle !== reminder.title || description !== reminder.description) {
      onUpdate({
        ...reminder,
        title: trimmedTitle,
        description,
      });
    }
  };

  const applyPresetDate = (hoursFromNow: number, targetHour?: number) => {
    const d = new Date(Date.now() + hoursFromNow * 3600000);
    if (typeof targetHour === 'number') {
      d.setHours(targetHour, 0, 0, 0);
    }
    const iso = d.toISOString();
    setDueDate(iso.slice(0, 16));
    onUpdate({ ...reminder, dueDate: iso });
  };

  const handleToggleChecklistItem = (itemId: string) => {
    const updated = reminder.checklist.map((item) =>
      item.id === itemId ? { ...item, completed: !item.completed } : item
    );
    onUpdate({ ...reminder, checklist: updated });
  };

  const handleAddChecklistItem = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newChecklistText.trim()) return;
    const newItem: ChecklistItem = {
      id: `chk-${Date.now()}`,
      text: newChecklistText.trim(),
      completed: false,
    };
    onUpdate({ ...reminder, checklist: [...reminder.checklist, newItem] });
    setNewChecklistText('');
  };

  const handleRemoveChecklistItem = (itemId: string) => {
    onUpdate({
      ...reminder,
      checklist: reminder.checklist.filter((item) => item.id !== itemId),
    });
  };

  const handleAddTag = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && newTagText.trim()) {
      e.preventDefault();
      const cleaned = newTagText.trim().replace(/^#/, '');
      if (cleaned && !reminder.tags.includes(cleaned)) {
        onUpdate({ ...reminder, tags: [...reminder.tags, cleaned] });
      }
      setNewTagText('');
    }
  };

  const handleRemoveTag = (tagToRemove: string) => {
    onUpdate({ ...reminder, tags: reminder.tags.filter((t) => t !== tagToRemove) });
  };

  const completedSteps = reminder.checklist.filter((c) => c.completed).length;

  const panelBody = (
    <div className="flex flex-col h-full bg-white dark:bg-zinc-950 xl:border-l border-zinc-200 dark:border-zinc-800/80 rounded-t-3xl xl:rounded-none overflow-hidden">
      {/* Mobile Drag Handle */}
      <div className="w-10 h-1.5 bg-zinc-300 dark:bg-zinc-700 rounded-full mx-auto mt-3 mb-1 xl:hidden shrink-0" />

      {/* Top Action Bar */}
      <div className="h-14 px-4 border-b border-zinc-200 dark:border-zinc-800/80 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => onUpdate({ ...reminder, completed: !reminder.completed })}
            className={`min-h-[40px] px-3 py-1.5 rounded-xl text-xs font-medium flex items-center gap-1.5 transition-colors whitespace-nowrap ${
              reminder.completed
                ? 'bg-emerald-50 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800/60'
                : 'bg-zinc-100 dark:bg-zinc-900 text-zinc-700 dark:text-zinc-200 hover:bg-zinc-200/70 dark:hover:bg-zinc-800'
            }`}
          >
            {reminder.completed ? (
              <CheckCircle2 className="w-4 h-4" />
            ) : (
              <Circle className="w-4 h-4" />
            )}
            <span>{reminder.completed ? 'Completed' : 'Mark done'}</span>
          </button>
        </div>

        <div className="flex items-center gap-0.5">
          {pushEnabled && onSendPushReminder && (
            <button
              onClick={() => onSendPushReminder(reminder)}
              className="min-h-[44px] min-w-[40px] flex items-center justify-center rounded-xl text-zinc-500 hover:text-indigo-600 dark:text-zinc-400 dark:hover:text-indigo-400 hover:bg-zinc-100 dark:hover:bg-zinc-900 transition-colors"
              title="Send push alert for this reminder"
            >
              <Send className="w-4 h-4" />
            </button>
          )}
          <button
            onClick={() => onUpdate({ ...reminder, starred: !reminder.starred })}
            className={`min-h-[44px] min-w-[40px] flex items-center justify-center rounded-xl transition-colors ${
              reminder.starred
                ? 'text-amber-500 bg-amber-50 dark:bg-amber-950/40'
                : 'text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-zinc-900'
            }`}
            title={reminder.starred ? 'Remove star' : 'Star reminder'}
          >
            <Star className={`w-4 h-4 ${reminder.starred ? 'fill-amber-500' : ''}`} />
          </button>
          <button
            onClick={() => onDelete(reminder.id)}
            className="min-h-[44px] min-w-[40px] flex items-center justify-center rounded-xl text-zinc-400 hover:text-rose-600 dark:hover:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-950/30 transition-colors"
            title="Delete reminder"
          >
            <Trash2 className="w-4 h-4" />
          </button>
          <button
            onClick={onClose}
            className="min-h-[44px] min-w-[40px] flex items-center justify-center rounded-xl text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-zinc-900 transition-colors"
            title="Close inspector"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Scrollable Properties & Content */}
      <div className="flex-1 overflow-y-auto p-5 space-y-6">
        {/* Inline Editable Title & Description */}
        <div className="space-y-2">
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            onBlur={commitTextFields}
            placeholder="Reminder title"
            className="w-full font-display text-lg font-semibold text-zinc-900 dark:text-zinc-100 bg-transparent border-0 p-0 focus:outline-none focus:ring-0"
          />
          <textarea
            rows={3}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            onBlur={commitTextFields}
            placeholder="Add notes, amounts, or extracted details..."
            className="w-full text-sm text-zinc-600 dark:text-zinc-400 bg-transparent border-0 p-0 resize-none focus:outline-none focus:ring-0 leading-relaxed"
          />
        </div>

        {/* Properties Grid */}
        <div className="border-t border-zinc-200 dark:border-zinc-800/80 pt-5 space-y-4">
          {/* Category */}
          <div className="flex items-center justify-between gap-4">
            <label className="text-xs font-medium text-zinc-500 dark:text-zinc-400 w-24 shrink-0">
              Category
            </label>
            <select
              value={reminder.category}
              onChange={(e) => onUpdate({ ...reminder, category: e.target.value as Category })}
              className="flex-1 h-8 px-2.5 bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg text-xs font-medium text-zinc-800 dark:text-zinc-200 focus:outline-none focus:border-indigo-500"
            >
              {CATEGORIES.map((cat) => (
                <option key={cat} value={cat}>
                  {cat}
                </option>
              ))}
            </select>
          </div>

          {/* Priority Segmented Control */}
          <div className="flex items-center justify-between gap-4">
            <label className="text-xs font-medium text-zinc-500 dark:text-zinc-400 w-24 shrink-0">
              Priority
            </label>
            <div className="flex-1 grid grid-cols-3 gap-1 p-1 bg-zinc-100 dark:bg-zinc-900 rounded-lg border border-zinc-200/60 dark:border-zinc-800">
              {PRIORITIES.map((p) => {
                const active = reminder.priority === p;
                return (
                  <button
                    key={p}
                    type="button"
                    onClick={() => onUpdate({ ...reminder, priority: p })}
                    className={`py-1 text-xs font-medium rounded-md capitalize transition-colors ${
                      active
                        ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-sm'
                        : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200'
                    }`}
                  >
                    {p}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Due Date & Quick Presets */}
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-4">
              <label className="text-xs font-medium text-zinc-500 dark:text-zinc-400 w-24 shrink-0">
                Due Date
              </label>
              <input
                type="datetime-local"
                value={dueDate}
                onChange={(e) => {
                  setDueDate(e.target.value);
                  if (e.target.value) {
                    onUpdate({ ...reminder, dueDate: new Date(e.target.value).toISOString() });
                  }
                }}
                className="flex-1 h-8 px-2.5 font-mono-nums bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg text-xs text-zinc-800 dark:text-zinc-200 focus:outline-none focus:border-indigo-500"
              />
            </div>
            <div className="flex items-center justify-end gap-1.5">
              <button
                type="button"
                onClick={() => applyPresetDate(0, 18)}
                className="px-2 py-1 text-[11px] font-medium text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 bg-zinc-100 dark:bg-zinc-900 rounded-md transition-colors"
              >
                Today 6 PM
              </button>
              <button
                type="button"
                onClick={() => applyPresetDate(24, 9)}
                className="px-2 py-1 text-[11px] font-medium text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 bg-zinc-100 dark:bg-zinc-900 rounded-md transition-colors"
              >
                Tomorrow 9 AM
              </button>
              <button
                type="button"
                onClick={() => applyPresetDate(168, 9)}
                className="px-2 py-1 text-[11px] font-medium text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 bg-zinc-100 dark:bg-zinc-900 rounded-md transition-colors"
              >
                Next Week
              </button>
            </div>
          </div>
        </div>

        {/* Checklist Section */}
        <div className="border-t border-zinc-200 dark:border-zinc-800/80 pt-5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
              Action Steps
            </span>
            {reminder.checklist.length > 0 && (
              <span className="font-mono-nums text-xs text-zinc-500 dark:text-zinc-400">
                {completedSteps}/{reminder.checklist.length}
              </span>
            )}
          </div>

          {reminder.checklist.length > 0 && (
            <div className="divide-y divide-zinc-100 dark:divide-zinc-800/60">
              {reminder.checklist.map((item) => (
                <div key={item.id} className="flex items-center justify-between py-2 group">
                  <button
                    type="button"
                    onClick={() => handleToggleChecklistItem(item.id)}
                    className="flex items-center gap-2.5 text-left flex-1 min-w-0"
                  >
                    {item.completed ? (
                      <CheckSquare className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
                    ) : (
                      <Square className="w-4 h-4 text-zinc-400 group-hover:text-zinc-600 dark:group-hover:text-zinc-300 shrink-0" />
                    )}
                    <span
                      className={`text-xs truncate ${
                        item.completed
                          ? 'line-through text-zinc-400 dark:text-zinc-500'
                          : 'text-zinc-800 dark:text-zinc-200'
                      }`}
                    >
                      {item.text}
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => handleRemoveChecklistItem(item.id)}
                    className="p-1 text-zinc-400 hover:text-rose-500 opacity-0 group-hover:opacity-100 transition-opacity"
                    aria-label="Remove step"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}

          <form onSubmit={handleAddChecklistItem} className="flex gap-2">
            <input
              type="text"
              value={newChecklistText}
              onChange={(e) => setNewChecklistText(e.target.value)}
              placeholder="Add an action step..."
              className="flex-1 h-8 px-3 bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg text-xs text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 focus:outline-none focus:border-indigo-500"
            />
            <button
              type="submit"
              className="px-3 h-8 bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 text-xs font-medium rounded-lg hover:opacity-90 transition-opacity flex items-center gap-1 whitespace-nowrap shrink-0"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Add</span>
            </button>
          </form>
        </div>

        {/* Tags Section (Unboxed metadata) */}
        <div className="border-t border-zinc-200 dark:border-zinc-800/80 pt-5 space-y-2.5">
          <label className="block text-xs font-medium text-zinc-500 dark:text-zinc-400">
            Tags
          </label>
          {reminder.tags.length > 0 && (
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-zinc-600 dark:text-zinc-300">
              {reminder.tags.map((tag) => (
                <span key={tag} className="inline-flex items-center gap-1 group">
                  <span>#{tag}</span>
                  <button
                    type="button"
                    onClick={() => handleRemoveTag(tag)}
                    className="text-zinc-400 hover:text-rose-500"
                    aria-label={`Remove tag ${tag}`}
                  >
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))}
            </div>
          )}
          <input
            type="text"
            value={newTagText}
            onChange={(e) => setNewTagText(e.target.value)}
            onKeyDown={handleAddTag}
            placeholder="Type tag and press Enter..."
            className="w-full h-8 px-3 bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-lg text-xs text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 focus:outline-none focus:border-indigo-500"
          />
        </div>

        {/* Screenshot Attachment & OCR Source */}
        {reminder.image && (
          <div className="border-t border-zinc-200 dark:border-zinc-800/80 pt-5 space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
                Source Scan{reminder.ocrEngine ? ` · ${reminder.ocrEngine}` : ''}
              </span>
              <button
                type="button"
                onClick={() => setLightboxOpen(true)}
                className="min-h-[36px] text-xs text-indigo-600 dark:text-indigo-400 hover:underline inline-flex items-center gap-1"
              >
                <Maximize2 className="w-3 h-3" /> Expand
              </button>
            </div>
            <div
              onClick={() => setLightboxOpen(true)}
              className="rounded-xl overflow-hidden border border-zinc-200 dark:border-zinc-800 bg-zinc-100 dark:bg-zinc-900 cursor-zoom-in"
            >
              <img
                src={reminder.image}
                alt={reminder.title}
                referrerPolicy="no-referrer"
                className="w-full max-h-56 object-contain"
              />
            </div>
            {reminder.ocrText && (
              <p className="text-xs text-zinc-500 dark:text-zinc-400 whitespace-pre-wrap leading-relaxed bg-zinc-50 dark:bg-zinc-900/60 border border-zinc-200/80 dark:border-zinc-800 rounded-xl p-3">
                {reminder.ocrText}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop Split-Pane Inspector */}
      <aside className="hidden xl:block w-[400px] shrink-0 h-screen sticky top-0">
        {panelBody}
      </aside>

      {/* Mobile / Tablet Slide-Up Bottom Sheet */}
      <div
        className="fixed inset-0 z-50 xl:hidden flex items-end justify-center bg-zinc-950/60 backdrop-blur-[2px]"
        onClick={onClose}
      >
        <div
          className="w-full max-w-xl max-h-[90vh] flex flex-col shadow-2xl"
          onClick={(e) => e.stopPropagation()}
        >
          {panelBody}
        </div>
      </div>

      {/* Screenshot Lightbox */}
      {lightboxOpen && reminder.image && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-black/85 backdrop-blur-sm"
          onClick={() => setLightboxOpen(false)}
        >
          <button
            onClick={() => setLightboxOpen(false)}
            className="absolute top-5 right-5 p-2 rounded-lg bg-zinc-900 text-white hover:bg-zinc-800"
            aria-label="Close lightbox"
          >
            <X className="w-5 h-5" />
          </button>
          <img
            src={reminder.image}
            alt={reminder.title}
            referrerPolicy="no-referrer"
            className="max-w-full max-h-[90vh] object-contain rounded-lg"
          />
        </div>
      )}
    </>
  );
};
