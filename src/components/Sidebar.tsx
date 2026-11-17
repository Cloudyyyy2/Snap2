import React from 'react';
import {
  Inbox,
  Calendar,
  Clock,
  CheckCircle,
  Star,
  Receipt,
  CalendarDays,
  CheckSquare,
  Plane,
  Briefcase,
  User,
  HeartPulse,
  Camera,
  Plus,
  BellRing,
  BellOff,
  Cloud,
  LogOut,
  Send,
  Sun,
  Moon,
  X,
} from 'lucide-react';
import { Category, Reminder } from '../types';

interface SidebarProps {
  currentFilter: string;
  setCurrentFilter: (filter: string) => void;
  selectedCategory: Category | 'All';
  setSelectedCategory: (cat: Category | 'All') => void;
  reminders: Reminder[];
  onOpenUpload: () => void;
  onOpenAdd: () => void;
  userEmail: string | null;
  onOpenAuth: () => void;
  onSignOut: () => void;
  pushEnabled: boolean;
  pushLoading: boolean;
  onTogglePush: () => void;
  onTestPush: () => void;
  isDark: boolean;
  onToggleTheme: () => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentFilter,
  setCurrentFilter,
  selectedCategory,
  setSelectedCategory,
  reminders,
  onOpenUpload,
  onOpenAdd,
  userEmail,
  onOpenAuth,
  onSignOut,
  pushEnabled,
  pushLoading,
  onTogglePush,
  onTestPush,
  isDark,
  onToggleTheme,
  mobileOpen,
  onCloseMobile,
}) => {
  const countByFilter = (filter: string) => {
    const todayStr = new Date().toISOString().split('T')[0];
    const nowIso = new Date().toISOString();

    return reminders.filter((r) => {
      if (filter === 'all') return !r.completed;
      if (filter === 'today') return !r.completed && r.dueDate.startsWith(todayStr);
      if (filter === 'upcoming') return !r.completed && r.dueDate > nowIso;
      if (filter === 'starred') return r.starred && !r.completed;
      if (filter === 'completed') return r.completed;
      return true;
    }).length;
  };

  const countByCategory = (cat: Category | 'All') => {
    if (cat === 'All') return reminders.filter((r) => !r.completed).length;
    return reminders.filter((r) => !r.completed && r.category === cat).length;
  };

  const views = [
    { id: 'all', label: 'Inbox', icon: <Inbox className="w-4 h-4" /> },
    { id: 'today', label: 'Today', icon: <Calendar className="w-4 h-4" /> },
    { id: 'upcoming', label: 'Upcoming', icon: <Clock className="w-4 h-4" /> },
    { id: 'starred', label: 'Starred', icon: <Star className="w-4 h-4" /> },
    { id: 'completed', label: 'Completed', icon: <CheckCircle className="w-4 h-4" /> },
  ];

  const categories: { name: Category; icon: React.ReactNode }[] = [
    { name: 'Bills', icon: <Receipt className="w-4 h-4 text-emerald-600 dark:text-emerald-400" /> },
    { name: 'Events', icon: <CalendarDays className="w-4 h-4 text-amber-600 dark:text-amber-400" /> },
    { name: 'Tasks', icon: <CheckSquare className="w-4 h-4 text-indigo-600 dark:text-indigo-400" /> },
    { name: 'Travel', icon: <Plane className="w-4 h-4 text-sky-600 dark:text-sky-400" /> },
    { name: 'Work', icon: <Briefcase className="w-4 h-4 text-violet-600 dark:text-violet-400" /> },
    { name: 'Personal', icon: <User className="w-4 h-4 text-rose-600 dark:text-rose-400" /> },
    { name: 'Health', icon: <HeartPulse className="w-4 h-4 text-teal-600 dark:text-teal-400" /> },
  ];

  const handleNavClick = (fn: () => void) => {
    fn();
    onCloseMobile();
  };

  const railContent = (
    <div className="flex flex-col h-full bg-zinc-50 dark:bg-zinc-950 border-r border-zinc-200 dark:border-zinc-800/80 select-none">
      {/* Brand Row */}
      <div className="h-14 px-4 flex items-center justify-between border-b border-zinc-200/80 dark:border-zinc-800/80 shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 flex items-center justify-center">
            <Camera className="w-4 h-4" />
          </div>
          <span className="font-display text-[15px] font-semibold tracking-tight text-zinc-900 dark:text-zinc-100 whitespace-nowrap">
            SnapRemind
          </span>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={onToggleTheme}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 hover:bg-zinc-200/60 dark:hover:bg-zinc-800/70 transition-colors"
            title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            aria-label="Toggle theme"
          >
            {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
          <button
            onClick={onCloseMobile}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 lg:hidden"
            aria-label="Close navigation"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Primary Capture Controls */}
      <div className="p-3 space-y-1.5">
        <button
          onClick={() => handleNavClick(onOpenUpload)}
          className="w-full flex items-center justify-between px-3 py-2 bg-indigo-600 hover:bg-indigo-700 dark:bg-indigo-600 dark:hover:bg-indigo-500 text-white text-xs font-medium rounded-lg transition-colors whitespace-nowrap"
        >
          <span className="flex items-center gap-2">
            <Camera className="w-3.5 h-3.5" />
            <span>Scan Screenshot</span>
          </span>
          <kbd className="font-mono-nums text-[10px] text-indigo-100/80">S</kbd>
        </button>

        <button
          onClick={() => handleNavClick(onOpenAdd)}
          className="w-full flex items-center justify-between px-3 py-2 bg-white dark:bg-zinc-900 hover:bg-zinc-100 dark:hover:bg-zinc-800/90 text-zinc-700 dark:text-zinc-200 border border-zinc-200 dark:border-zinc-800 text-xs font-medium rounded-lg transition-colors whitespace-nowrap"
        >
          <span className="flex items-center gap-2">
            <Plus className="w-3.5 h-3.5 text-zinc-500" />
            <span>New Reminder</span>
          </span>
          <kbd className="font-mono-nums text-[10px] text-zinc-400">N</kbd>
        </button>
      </div>

      {/* Views & Categories */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-6">
        <div>
          <div className="px-2.5 mb-1.5 text-[11px] font-medium text-zinc-400 dark:text-zinc-500">
            Views
          </div>
          <nav className="space-y-0.5">
            {views.map((item) => {
              const active = currentFilter === item.id;
              const count = countByFilter(item.id);
              return (
                <button
                  key={item.id}
                  onClick={() => handleNavClick(() => setCurrentFilter(item.id))}
                  className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-[13px] transition-colors whitespace-nowrap ${
                    active
                      ? 'bg-zinc-200/80 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 font-medium'
                      : 'text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200/50 dark:hover:bg-zinc-900 hover:text-zinc-900 dark:hover:text-zinc-200'
                  }`}
                >
                  <span className="flex items-center gap-2.5">
                    <span className={active ? 'text-indigo-600 dark:text-indigo-400' : 'text-zinc-400 dark:text-zinc-500'}>
                      {item.icon}
                    </span>
                    <span>{item.label}</span>
                  </span>
                  {count > 0 && (
                    <span className="font-mono-nums text-[11px] text-zinc-500 dark:text-zinc-400">
                      {count}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        <div>
          <div className="px-2.5 mb-1.5 flex items-center justify-between">
            <span className="text-[11px] font-medium text-zinc-400 dark:text-zinc-500">
              Categories
            </span>
            {selectedCategory !== 'All' && (
              <button
                onClick={() => setSelectedCategory('All')}
                className="text-[11px] text-indigo-600 dark:text-indigo-400 hover:underline"
              >
                All
              </button>
            )}
          </div>
          <nav className="space-y-0.5">
            {categories.map((cat) => {
              const active = selectedCategory === cat.name;
              const count = countByCategory(cat.name);
              return (
                <button
                  key={cat.name}
                  onClick={() =>
                    handleNavClick(() =>
                      setSelectedCategory(active ? 'All' : cat.name)
                    )
                  }
                  className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-[13px] transition-colors whitespace-nowrap ${
                    active
                      ? 'bg-zinc-200/80 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 font-medium'
                      : 'text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200/50 dark:hover:bg-zinc-900 hover:text-zinc-900 dark:hover:text-zinc-200'
                  }`}
                >
                  <span className="flex items-center gap-2.5">
                    {cat.icon}
                    <span>{cat.name}</span>
                  </span>
                  {count > 0 && (
                    <span className="font-mono-nums text-[11px] text-zinc-500 dark:text-zinc-400">
                      {count}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Bottom Workspace Utilities: Push Alerts & Supabase Cloud Sync */}
      <div className="p-3 border-t border-zinc-200 dark:border-zinc-800/80 space-y-1.5 shrink-0">
        <div className="flex items-center gap-1.5">
          <button
            onClick={onTogglePush}
            disabled={pushLoading}
            className={`flex-1 flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors whitespace-nowrap ${
              pushEnabled
                ? 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800/60'
                : 'text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200/60 dark:hover:bg-zinc-900 border border-transparent'
            }`}
          >
            <span className="flex items-center gap-2">
              {pushEnabled ? <BellRing className="w-3.5 h-3.5" /> : <BellOff className="w-3.5 h-3.5" />}
              <span>{pushEnabled ? 'Push Alerts On' : 'Enable Push'}</span>
            </span>
          </button>

          {pushEnabled && (
            <button
              onClick={onTestPush}
              className="p-1.5 rounded-lg border border-zinc-200 dark:border-zinc-800 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-200/60 dark:hover:bg-zinc-800 transition-colors"
              title="Send test push notification"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {userEmail ? (
          <button
            onClick={onSignOut}
            className="w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200/60 dark:hover:bg-zinc-900 transition-colors"
            title={`Signed in as ${userEmail} — Click to sign out`}
          >
            <span className="flex items-center gap-2 min-w-0">
              <Cloud className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 shrink-0" />
              <span className="truncate">{userEmail}</span>
            </span>
            <LogOut className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
          </button>
        ) : (
          <button
            onClick={() => handleNavClick(onOpenAuth)}
            className="w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs font-medium text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200/60 dark:hover:bg-zinc-900 hover:text-zinc-900 dark:hover:text-zinc-200 transition-colors whitespace-nowrap"
          >
            <span className="flex items-center gap-2">
              <Cloud className="w-3.5 h-3.5 text-zinc-400" />
              <span>Cloud Sync Sign In</span>
            </span>
          </button>
        )}
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop Fixed Rail */}
      <aside className="hidden lg:block w-[248px] shrink-0 h-screen sticky top-0">
        {railContent}
      </aside>

      {/* Mobile Drawer */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 lg:hidden bg-zinc-950/50 backdrop-blur-[2px]"
          onClick={onCloseMobile}
        >
          <aside
            className="w-[272px] h-full"
            onClick={(e) => e.stopPropagation()}
          >
            {railContent}
          </aside>
        </div>
      )}
    </>
  );
};
