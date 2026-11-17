import React from 'react';
import {
  CheckCircle2,
  Circle,
  Star,
  Paperclip,
  Trash2,
} from 'lucide-react';
import { Reminder } from '../types';

interface ReminderCardProps {
  reminder: Reminder;
  variant?: 'row' | 'card';
  isSelected?: boolean;
  onToggleComplete: (id: string) => void;
  onToggleStar: (id: string) => void;
  onDelete?: (id: string) => void;
  onClick: () => void;
}

export const ReminderCard: React.FC<ReminderCardProps> = ({
  reminder,
  variant = 'row',
  isSelected = false,
  onToggleComplete,
  onToggleStar,
  onDelete,
  onClick,
}) => {
  const isOverdue = !reminder.completed && new Date(reminder.dueDate).getTime() < Date.now();
  const isToday =
    !reminder.completed &&
    new Date(reminder.dueDate).toDateString() === new Date().toDateString();

  const dueDateObj = new Date(reminder.dueDate);
  const formattedDate = dueDateObj.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
  });
  const formattedTime = dueDateObj.toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit',
  });

  const completedSteps = reminder.checklist.filter((c) => c.completed).length;
  const totalSteps = reminder.checklist.length;
  const progressPct = totalSteps > 0 ? (completedSteps / totalSteps) * 100 : 0;

  const priorityLabel =
    reminder.priority === 'high'
      ? 'High priority'
      : reminder.priority === 'low'
        ? 'Low priority'
        : 'Medium';

  if (variant === 'row') {
    return (
      <div
        onClick={onClick}
        className={`group flex items-center justify-between gap-2 px-3 sm:px-4 py-2.5 min-h-[60px] border-b border-zinc-200/80 dark:border-zinc-800/80 cursor-pointer transition-colors ${
          isSelected
            ? 'bg-indigo-50/70 dark:bg-indigo-950/30'
            : 'bg-white dark:bg-zinc-950 hover:bg-zinc-50 dark:hover:bg-zinc-900/70'
        } ${reminder.completed ? 'opacity-60' : ''}`}
      >
        {/* Left: Checkbox + Title + Unboxed Metadata */}
        <div className="flex items-center gap-1.5 sm:gap-2.5 min-w-0 flex-1">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onToggleComplete(reminder.id);
            }}
            className={`min-h-[44px] min-w-[40px] -ml-1.5 flex items-center justify-center shrink-0 transition-colors ${
              reminder.completed
                ? 'text-emerald-600 dark:text-emerald-400'
                : 'text-zinc-300 dark:text-zinc-600 hover:text-indigo-600 dark:hover:text-indigo-400'
            }`}
            aria-label={reminder.completed ? 'Mark as active' : 'Mark as complete'}
          >
            {reminder.completed ? (
              <CheckCircle2 className="w-5 h-5 sm:w-4 sm:h-4" />
            ) : (
              <Circle className="w-5 h-5 sm:w-4 sm:h-4" />
            )}
          </button>

          <div className="min-w-0 flex-1 py-0.5">
            <div className="flex items-center gap-2">
              <h3
                className={`text-sm font-medium truncate ${
                  reminder.completed
                    ? 'line-through text-zinc-400 dark:text-zinc-500'
                    : 'text-zinc-900 dark:text-zinc-100'
                }`}
              >
                {reminder.title}
              </h3>
              {reminder.image && (
                <Paperclip
                  className="w-3.5 h-3.5 text-zinc-400 shrink-0"
                  aria-label="Has screenshot"
                />
              )}
            </div>

            <div className="flex items-center flex-wrap gap-x-2 gap-y-0.5 mt-0.5 text-xs text-zinc-500 dark:text-zinc-400">
              <span>{reminder.category}</span>
              <span aria-hidden="true">·</span>
              <span
                className={`font-mono-nums ${
                  isOverdue
                    ? 'text-rose-600 dark:text-rose-400 font-medium'
                    : isToday
                      ? 'text-amber-600 dark:text-amber-400 font-medium'
                      : ''
                }`}
              >
                {isOverdue ? `Overdue ${formattedDate}` : isToday ? `Today ${formattedTime}` : `${formattedDate}, ${formattedTime}`}
              </span>
              {totalSteps > 0 && (
                <>
                  <span aria-hidden="true">·</span>
                  <span className="font-mono-nums">
                    {completedSteps}/{totalSteps} steps
                  </span>
                </>
              )}
              {reminder.priority === 'high' && !reminder.completed && (
                <>
                  <span aria-hidden="true">·</span>
                  <span className="text-rose-600 dark:text-rose-400 font-medium">
                    High priority
                  </span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Right: Touch-friendly Star & Delete */}
        <div
          className="flex items-center shrink-0"
          onClick={(e) => e.stopPropagation()}
        >
          <button
            type="button"
            onClick={() => onToggleStar(reminder.id)}
            className={`min-h-[44px] min-w-[40px] flex items-center justify-center rounded-lg transition-colors ${
              reminder.starred
                ? 'text-amber-500'
                : 'text-zinc-300 dark:text-zinc-600 lg:opacity-0 lg:group-hover:opacity-100 hover:text-zinc-600 dark:hover:text-zinc-300'
            }`}
            aria-label={reminder.starred ? 'Unstar reminder' : 'Star reminder'}
          >
            <Star className={`w-4 h-4 ${reminder.starred ? 'fill-amber-500' : ''}`} />
          </button>

          {onDelete && (
            <button
              type="button"
              onClick={() => onDelete(reminder.id)}
              className="hidden sm:flex min-h-[44px] min-w-[40px] items-center justify-center rounded-lg text-zinc-400 opacity-0 group-hover:opacity-100 hover:text-rose-600 dark:hover:text-rose-400 transition-opacity"
              aria-label="Delete reminder"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
    );
  }

  // Card variant (Grid & Board views) — Single-elevation, zero-pill metadata
  return (
    <div
      onClick={onClick}
      className={`group rounded-xl border p-4 cursor-pointer transition-colors flex flex-col justify-between gap-3 ${
        isSelected
          ? 'bg-indigo-50/50 dark:bg-indigo-950/25 border-indigo-400 dark:border-indigo-700'
          : 'bg-white dark:bg-zinc-900/60 border-zinc-200 dark:border-zinc-800 hover:border-zinc-300 dark:hover:border-zinc-700'
      } ${reminder.completed ? 'opacity-65' : ''}`}
    >
      <div className="space-y-2.5">
        {/* Quiet 1-line kicker + action icons */}
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 text-xs text-zinc-500 dark:text-zinc-400 truncate">
            <span className="font-medium text-zinc-700 dark:text-zinc-300">{reminder.category}</span>
            <span aria-hidden="true">·</span>
            <span className={reminder.priority === 'high' && !reminder.completed ? 'text-rose-600 dark:text-rose-400 font-medium' : ''}>
              {priorityLabel}
            </span>
          </div>

          <div className="flex items-center gap-1 shrink-0" onClick={(e) => e.stopPropagation()}>
            <button
              type="button"
              onClick={() => onToggleStar(reminder.id)}
              className={`p-1 rounded-md transition-colors ${
                reminder.starred
                  ? 'text-amber-500'
                  : 'text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200'
              }`}
              aria-label={reminder.starred ? 'Unstar' : 'Star'}
            >
              <Star className={`w-3.5 h-3.5 ${reminder.starred ? 'fill-amber-500' : ''}`} />
            </button>
            <button
              type="button"
              onClick={() => onToggleComplete(reminder.id)}
              className={`p-1 rounded-md transition-colors ${
                reminder.completed
                  ? 'text-emerald-600 dark:text-emerald-400'
                  : 'text-zinc-400 hover:text-indigo-600 dark:hover:text-indigo-400'
              }`}
              aria-label={reminder.completed ? 'Mark active' : 'Mark complete'}
            >
              {reminder.completed ? <CheckCircle2 className="w-4 h-4" /> : <Circle className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Title & Notes */}
        <div>
          <h3
            className={`font-display text-sm font-semibold leading-snug ${
              reminder.completed
                ? 'line-through text-zinc-400 dark:text-zinc-500'
                : 'text-zinc-900 dark:text-zinc-100'
            }`}
          >
            {reminder.title}
          </h3>
          {reminder.description && (
            <p className="text-xs text-zinc-500 dark:text-zinc-400 line-clamp-2 mt-1 leading-relaxed">
              {reminder.description}
            </p>
          )}
        </div>

        {/* Checklist progress hairline */}
        {totalSteps > 0 && (
          <div className="space-y-1.5 pt-1">
            <div className="flex items-center justify-between text-[11px] text-zinc-500 dark:text-zinc-400">
              <span>Steps</span>
              <span className="font-mono-nums">
                {completedSteps}/{totalSteps}
              </span>
            </div>
            <div className="w-full h-1 bg-zinc-100 dark:bg-zinc-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-indigo-600 dark:bg-indigo-500 transition-all duration-200"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>
        )}

        {/* Attached Screenshot Thumbnail */}
        {reminder.image && (
          <div className="rounded-lg overflow-hidden border border-zinc-200 dark:border-zinc-800 h-28 bg-zinc-50 dark:bg-zinc-950">
            <img
              src={reminder.image}
              alt={reminder.title}
              referrerPolicy="no-referrer"
              className="w-full h-full object-cover"
            />
          </div>
        )}
      </div>

      {/* Footer Metadata (Unboxed text) */}
      <div className="pt-2.5 border-t border-zinc-100 dark:border-zinc-800/80 flex items-center justify-between gap-2 text-xs">
        <span
          className={`font-mono-nums ${
            isOverdue
              ? 'text-rose-600 dark:text-rose-400 font-medium'
              : isToday
                ? 'text-amber-600 dark:text-amber-400 font-medium'
                : 'text-zinc-500 dark:text-zinc-400'
          }`}
        >
          {isOverdue ? `Overdue · ${formattedDate}` : isToday ? `Today · ${formattedTime}` : `${formattedDate} · ${formattedTime}`}
        </span>

        {reminder.tags.length > 0 && (
          <span className="text-[11px] text-zinc-400 dark:text-zinc-500 truncate">
            {reminder.tags.slice(0, 2).map((t) => `#${t}`).join(' ')}
          </span>
        )}
      </div>
    </div>
  );
};
