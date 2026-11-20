import React, { useState, useEffect, useCallback } from 'react';
import { Navbar } from './components/Navbar';
import { Sidebar } from './components/Sidebar';
import { BottomNav, MobileTab } from './components/BottomNav';
import { OcrStudio } from './components/OcrStudio';
import { ReminderCard } from './components/ReminderCard';
import { ReminderInspector } from './components/ReminderInspector';
import { AddReminderModal } from './components/AddReminderModal';
import { AuthModal } from './components/AuthModal';
import { CommandPalette } from './components/CommandPalette';
import { Reminder, Category } from './types';
import {
  getStoredReminders,
  saveStoredReminders,
  fetchSupabaseReminders,
  createSupabaseReminder,
  updateSupabaseReminder,
  deleteSupabaseReminder,
} from './services/storage';
import { getSupabaseClient } from './lib/supabase';
import {
  getExistingPushSubscription,
  subscribeToWebPush,
  unsubscribeFromWebPush,
  sendTestPushNotification,
} from './services/push';
import {
  Plus,
  Camera,
  List,
  LayoutGrid,
  Columns3,
  ArrowUpDown,
  Loader2,
  Search,
  X,
} from 'lucide-react';

type ViewMode = 'list' | 'grid' | 'board';

export default function App() {
  const [reminders, setReminders] = useState<Reminder[]>([]);
  const [currentFilter, setCurrentFilter] = useState<string>('scan');
  const [selectedCategory, setSelectedCategory] = useState<Category | 'All'>('All');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [sortBy, setSortBy] = useState<'dueDate' | 'priority' | 'title'>('dueDate');
  const [viewMode, setViewMode] = useState<ViewMode>('list');

  const [activeReminder, setActiveReminder] = useState<Reminder | null>(null);
  const [droppedImage, setDroppedImage] = useState<string | null>(null);
  const [reviewSheetOpen, setReviewSheetOpen] = useState(false);
  const [isAddOpen, setIsAddOpen] = useState(false);
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [isCommandOpen, setIsCommandOpen] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  const [quickInput, setQuickInput] = useState('');
  const [quickParsing, setQuickParsing] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);

  const [isDark, setIsDark] = useState<boolean>(() => {
    try {
      const saved = localStorage.getItem('snapremind_theme');
      return saved === 'dark';
    } catch {
      return false;
    }
  });

  const [userId, setUserId] = useState<string | null>(null);
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const [pushEnabled, setPushEnabled] = useState(false);
  const [pushLoading, setPushLoading] = useState(false);
  const [toast, setToast] = useState<{
    message: string;
    type: 'success' | 'error';
    actionLabel?: string;
    onAction?: () => void;
  } | null>(null);

  useEffect(() => {
    const root = document.documentElement;
    if (isDark) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    try {
      localStorage.setItem('snapremind_theme', isDark ? 'dark' : 'light');
    } catch {
      // ignore storage failure
    }
  }, [isDark]);

  const showToast = useCallback(
    (
      message: string,
      type: 'success' | 'error' = 'success',
      action?: { label: string; onClick: () => void }
    ) => {
      setToast({
        message,
        type,
        actionLabel: action?.label,
        onAction: action?.onClick,
      });
      setTimeout(() => {
        setToast((prev) => (prev?.message === message ? null : prev));
      }, 4500);
    },
    []
  );

  const loadReminders = useCallback(
    async (uid: string | null) => {
      if (uid) {
        try {
          const cloudReminders = await fetchSupabaseReminders(uid);
          setReminders(cloudReminders);
          return;
        } catch (err: any) {
          console.error('Failed to load cloud reminders:', err);
          showToast(err?.message || 'Could not load cloud reminders, showing local reminders.', 'error');
        }
      }
      setReminders(getStoredReminders());
    },
    [showToast]
  );

  useEffect(() => {
    let subscriptionCleanup: (() => void) | undefined;

    const init = async () => {
      try {
        const existingSub = await getExistingPushSubscription();
        setPushEnabled(Boolean(existingSub));
      } catch {
        setPushEnabled(false);
      }

      const supabase = await getSupabaseClient();
      if (supabase) {
        const { data } = await supabase.auth.getSession();
        const sessionUser = data.session?.user || null;
        setUserId(sessionUser?.id || null);
        setUserEmail(sessionUser?.email || null);
        await loadReminders(sessionUser?.id || null);

        const { data: authListener } = supabase.auth.onAuthStateChange((_event, session) => {
          const u = session?.user || null;
          setUserId(u?.id || null);
          setUserEmail(u?.email || null);
          loadReminders(u?.id || null);
        });
        subscriptionCleanup = () => authListener.subscription.unsubscribe();
      } else {
        setReminders(getStoredReminders());
      }
    };

    init();

    return () => {
      if (subscriptionCleanup) subscriptionCleanup();
    };
  }, [loadReminders]);

  // Global keyboard shortcuts (Cmd+K, N, S, Escape)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const isTyping =
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable);

      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setIsCommandOpen((prev) => !prev);
        return;
      }

      if (e.key === 'Escape') {
        if (isCommandOpen) setIsCommandOpen(false);
        else if (isAddOpen) setIsAddOpen(false);
        else if (isAuthOpen) setIsAuthOpen(false);
        else if (activeReminder) setActiveReminder(null);
        return;
      }

      if (isTyping || isCommandOpen || isAddOpen || isAuthOpen) return;

      if (e.key.toLowerCase() === 'n') {
        e.preventDefault();
        setIsAddOpen(true);
      } else if (e.key.toLowerCase() === 's') {
        e.preventDefault();
        setCurrentFilter('scan');
        setActiveReminder(null);
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isCommandOpen, isAddOpen, isAuthOpen, activeReminder]);

  const updateLocalState = (updatedList: Reminder[]) => {
    setReminders(updatedList);
    if (!userId) {
      saveStoredReminders(updatedList);
    }
  };

  const handleToggleComplete = async (id: string) => {
    const target = reminders.find((r) => r.id === id);
    if (!target) return;
    const updatedReminder = { ...target, completed: !target.completed };
    const updatedList = reminders.map((r) => (r.id === id ? updatedReminder : r));
    updateLocalState(updatedList);
    if (activeReminder?.id === id) setActiveReminder(updatedReminder);

    if (userId) {
      try {
        await updateSupabaseReminder(userId, updatedReminder);
      } catch (err: any) {
        showToast(err?.message || 'Failed to sync completion status.', 'error');
      }
    }
  };

  const handleToggleStar = async (id: string) => {
    const target = reminders.find((r) => r.id === id);
    if (!target) return;
    const updatedReminder = { ...target, starred: !target.starred };
    const updatedList = reminders.map((r) => (r.id === id ? updatedReminder : r));
    updateLocalState(updatedList);
    if (activeReminder?.id === id) setActiveReminder(updatedReminder);

    if (userId) {
      try {
        await updateSupabaseReminder(userId, updatedReminder);
      } catch (err: any) {
        showToast(err?.message || 'Failed to sync star status.', 'error');
      }
    }
  };

  const handleDeleteReminder = async (id: string) => {
    const updatedList = reminders.filter((r) => r.id !== id);
    updateLocalState(updatedList);
    if (activeReminder?.id === id) {
      setActiveReminder(null);
    }

    if (userId) {
      try {
        await deleteSupabaseReminder(userId, id);
      } catch (err: any) {
        showToast(err?.message || 'Failed to delete cloud reminder.', 'error');
      }
    }
  };

  const handleUpdateReminder = async (updatedReminder: Reminder) => {
    const updatedList = reminders.map((r) =>
      r.id === updatedReminder.id ? updatedReminder : r
    );
    updateLocalState(updatedList);
    setActiveReminder(updatedReminder);

    if (userId) {
      try {
        await updateSupabaseReminder(userId, updatedReminder);
      } catch (err: any) {
        showToast(err?.message || 'Failed to update cloud reminder.', 'error');
      }
    }
  };

  const handleAddReminder = async (newReminder: Reminder) => {
    if (userId) {
      try {
        const saved = await createSupabaseReminder(userId, newReminder);
        setReminders((prev) => [saved, ...prev]);
        if (currentFilter !== 'scan') {
          setActiveReminder(saved);
        }
        showToast('Reminder saved.', 'success', {
          label: 'View',
          onClick: () => {
            setCurrentFilter('all');
            setActiveReminder(saved);
          },
        });
        return;
      } catch (err: any) {
        showToast(err?.message || 'Failed to save to cloud; saved locally instead.', 'error');
      }
    }

    const updatedList = [newReminder, ...reminders];
    updateLocalState(updatedList);
    if (currentFilter !== 'scan') {
      setActiveReminder(newReminder);
    }
    showToast('Reminder saved.', 'success', {
      label: 'View',
      onClick: () => {
        setCurrentFilter('all');
        setActiveReminder(newReminder);
      },
    });
  };

  const handleQuickCapture = async (useAI: boolean) => {
    const text = quickInput.trim();
    if (!text) return;

    if (!useAI) {
      const quickReminder: Reminder = {
        id: `rem-${Date.now()}`,
        title: text,
        description: '',
        dueDate: new Date(Date.now() + 86400000).toISOString(),
        category: selectedCategory !== 'All' ? selectedCategory : 'Tasks',
        priority: 'medium',
        checklist: [],
        tags: ['quick'],
        completed: false,
        starred: false,
        createdAt: new Date().toISOString(),
      };
      setQuickInput('');
      await handleAddReminder(quickReminder);
      return;
    }

    setQuickParsing(true);
    try {
      const res = await fetch('/api/extract-reminder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text }),
      });
      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || 'Failed to parse reminder');
      }
      const parsed = data.data || {};
      const validCategories = ['Bills', 'Events', 'Tasks', 'Travel', 'Work', 'Personal', 'Health'];
      const validPriorities = ['high', 'medium', 'low'];
      const newReminder: Reminder = {
        id: `rem-${Date.now()}`,
        title: parsed.title || text,
        description: parsed.description || '',
        dueDate:
          parsed.dueDate && !isNaN(new Date(parsed.dueDate).getTime())
            ? new Date(parsed.dueDate).toISOString()
            : new Date(Date.now() + 86400000).toISOString(),
        category: validCategories.includes(parsed.category) ? parsed.category : 'Tasks',
        priority: validPriorities.includes(parsed.priority) ? parsed.priority : 'medium',
        checklist: Array.isArray(parsed.checklist)
          ? parsed.checklist.map((c: any, idx: number) => ({
              id: `chk-${Date.now()}-${idx}`,
              text: typeof c === 'string' ? c : c?.text || '',
              completed: Boolean(c?.completed),
            }))
          : [],
        tags: Array.isArray(parsed.tags) ? parsed.tags : ['ai'],
        completed: false,
        starred: false,
        createdAt: new Date().toISOString(),
      };
      setQuickInput('');
      await handleAddReminder(newReminder);
    } catch (err: any) {
      showToast(err?.message || 'Could not parse reminder with AI.', 'error');
    } finally {
      setQuickParsing(false);
    }
  };

  const handleDropOnWorkspace = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const result = ev.target?.result as string;
      if (result) {
        setCurrentFilter('scan');
        setActiveReminder(null);
        setDroppedImage(result);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleSignOut = async () => {
    const supabase = await getSupabaseClient();
    if (supabase) {
      await supabase.auth.signOut();
      showToast('Signed out of cloud sync.');
    }
  };

  const handleTogglePush = async () => {
    setPushLoading(true);
    try {
      if (pushEnabled) {
        await unsubscribeFromWebPush(userId || undefined);
        setPushEnabled(false);
        showToast('Push notifications disabled.');
      } else {
        await subscribeToWebPush(userId || undefined);
        setPushEnabled(true);
        showToast('Push notifications enabled.');
      }
    } catch (err: any) {
      showToast(err?.message || 'Could not update push notification settings.', 'error');
    } finally {
      setPushLoading(false);
    }
  };

  const handleTestPush = async () => {
    try {
      await sendTestPushNotification();
      showToast('Test push notification sent.');
    } catch (err: any) {
      showToast(err?.message || 'Failed to send test push notification.', 'error');
    }
  };

  const handleSendPushForReminder = async (reminder: Reminder) => {
    try {
      const res = await fetch('/api/push/send-test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: reminder.title,
          body: reminder.description || `Due ${new Date(reminder.dueDate).toLocaleString()}`,
        }),
      });
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.error || 'Push failed');
      showToast(`Push alert sent for "${reminder.title}".`);
    } catch (err: any) {
      showToast(err?.message || 'Could not send push alert.', 'error');
    }
  };

  // Filter & Search logic
  const todayStr = new Date().toISOString().split('T')[0];
  const nowIso = new Date().toISOString();

  const filteredReminders = reminders.filter((r) => {
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const matchTitle = r.title.toLowerCase().includes(q);
      const matchDesc = r.description.toLowerCase().includes(q);
      const matchOcr = (r.ocrText || '').toLowerCase().includes(q);
      const matchTags = r.tags.some((t) => t.toLowerCase().includes(q));
      if (!matchTitle && !matchDesc && !matchOcr && !matchTags) return false;
    }

    if (selectedCategory !== 'All' && currentFilter !== 'search' && r.category !== selectedCategory) {
      return false;
    }

    if (currentFilter === 'all') return !r.completed;
    if (currentFilter === 'today') return !r.completed && r.dueDate.startsWith(todayStr);
    if (currentFilter === 'upcoming') return !r.completed && r.dueDate > nowIso;
    if (currentFilter === 'starred') return r.starred && !r.completed;
    if (currentFilter === 'completed') return r.completed;
    if (currentFilter === 'search') return true;

    return !r.completed;
  });

  const sortedReminders = [...filteredReminders].sort((a, b) => {
    if (sortBy === 'dueDate') {
      return new Date(a.dueDate).getTime() - new Date(b.dueDate).getTime();
    }
    if (sortBy === 'priority') {
      const weights: Record<string, number> = { high: 3, medium: 2, low: 1 };
      return (weights[b.priority] || 0) - (weights[a.priority] || 0);
    }
    return a.title.localeCompare(b.title);
  });

  const activeCount = reminders.filter((r) => !r.completed).length;
  const todayCount = reminders.filter((r) => !r.completed && r.dueDate.startsWith(todayStr)).length;
  const todayEndIso = `${todayStr}T23:59:59.999Z`;
  const todaySpotlights = reminders.filter((r) => !r.completed && r.dueDate <= todayEndIso);

  // Sync active reminders to host so background random-window push scheduler works even when tab is closed
  useEffect(() => {
    const payload = reminders.map((r) => ({
      id: r.id,
      title: r.title,
      dueDate: r.dueDate,
      completed: r.completed,
      priority: r.priority,
    }));
    fetch('/api/reminders/sync', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reminders: payload }),
    }).catch(() => {});
  }, [reminders]);

  // Showcase a random Today's Reminder nudge when the user opens or pulls up the app
  useEffect(() => {
    const triggerDailyNudge = () => {
      if (todaySpotlights.length === 0) return;
      const lastNudgeAt = Number(sessionStorage.getItem('snapremind_last_nudge') || '0');
      if (Date.now() - lastNudgeAt < 1000 * 60 * 15) return; // at most once per 15m per tab session
      sessionStorage.setItem('snapremind_last_nudge', String(Date.now()));
      const pick = todaySpotlights[Math.floor(Math.random() * todaySpotlights.length)];
      showToast(`Today's Reminder: ${pick.title}`, 'success', {
        label: 'View',
        onClick: () => setActiveReminder(pick),
      });
      if (pushEnabled) {
        fetch('/api/push/nudge', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            title: "Today's Reminder",
            body: pick.title,
          }),
        }).catch(() => {});
      }
    };

    triggerDailyNudge();
    const onVis = () => {
      if (document.visibilityState === 'visible') triggerDailyNudge();
    };
    document.addEventListener('visibilitychange', onVis);
    return () => document.removeEventListener('visibilitychange', onVis);
  }, [todaySpotlights.length, pushEnabled]);

  // Board columns
  const urgentOrOverdue = sortedReminders.filter(
    (r) => !r.completed && (r.priority === 'high' || new Date(r.dueDate).getTime() < Date.now())
  );
  const scheduled = sortedReminders.filter(
    (r) => !r.completed && r.priority !== 'high' && new Date(r.dueDate).getTime() >= Date.now()
  );
  const doneList = reminders.filter((r) => r.completed);

  const activeMobileTab: MobileTab =
    currentFilter === 'scan'
      ? 'scan'
      : currentFilter === 'today'
        ? 'today'
        : currentFilter === 'search'
          ? 'search'
          : 'inbox';

  const handleMobileTabSelect = (tab: MobileTab) => {
    setActiveReminder(null);
    if (tab === 'scan') setCurrentFilter('scan');
    else if (tab === 'inbox') setCurrentFilter('all');
    else if (tab === 'today') setCurrentFilter('today');
    else if (tab === 'search') setCurrentFilter('search');
  };

  const hideMobileBottomNav =
    reviewSheetOpen || Boolean(activeReminder) || isAddOpen || isAuthOpen || mobileNavOpen;

  return (
    <div
      className="min-h-screen bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 flex selection:bg-indigo-600 selection:text-white"
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragOver(true);
      }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={handleDropOnWorkspace}
    >
      {/* Left Navigation Rail (Desktop fixed / Mobile drawer for Categories, Push & Cloud Sync) */}
      <Sidebar
        currentFilter={currentFilter}
        setCurrentFilter={(f) => {
          setCurrentFilter(f);
          if (f === 'scan') setActiveReminder(null);
        }}
        selectedCategory={selectedCategory}
        setSelectedCategory={(cat) => {
          setSelectedCategory(cat);
          if (currentFilter === 'scan' || currentFilter === 'search') {
            setCurrentFilter('all');
          }
        }}
        reminders={reminders}
        onOpenUpload={() => {
          setCurrentFilter('scan');
          setActiveReminder(null);
        }}
        onOpenAdd={() => setIsAddOpen(true)}
        userEmail={userEmail}
        onOpenAuth={() => setIsAuthOpen(true)}
        onSignOut={handleSignOut}
        pushEnabled={pushEnabled}
        pushLoading={pushLoading}
        onTogglePush={handleTogglePush}
        onTestPush={handleTestPush}
        isDark={isDark}
        onToggleTheme={() => setIsDark((d) => !d)}
        mobileOpen={mobileNavOpen}
        onCloseMobile={() => setMobileNavOpen(false)}
      />

      {/* Center Workspace Column */}
      <div className="flex-1 min-w-0 flex flex-col">
        <Navbar
          currentFilter={currentFilter}
          selectedCategory={selectedCategory}
          activeCount={activeCount}
          searchQuery={searchQuery}
          setSearchQuery={(q) => {
            setSearchQuery(q);
            if (q && currentFilter === 'scan') {
              setCurrentFilter('search');
            }
          }}
          onOpenCommandPalette={() => setIsCommandOpen(true)}
          onOpenScan={() => {
            setCurrentFilter('scan');
            setActiveReminder(null);
          }}
          onOpenMobileMenu={() => setMobileNavOpen(true)}
        />

        {currentFilter === 'scan' ? (
          <main className="flex-1">
            <OcrStudio
              onSaveReminder={handleAddReminder}
              externalImage={droppedImage}
              onClearExternalImage={() => setDroppedImage(null)}
              onSheetOpenChange={setReviewSheetOpen}
            />
          </main>
        ) : (
          <main className="flex-1 px-4 lg:px-8 pt-4 pb-24 lg:py-6 max-w-5xl w-full mx-auto space-y-4">
            {/* Mobile-First Search Bar when in Search Tab */}
            {currentFilter === 'search' && (
              <div className="relative">
                <Search className="w-4 h-4 text-zinc-400 absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                <input
                  type="text"
                  autoFocus
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search titles, notes, OCR text, or tags..."
                  className="w-full min-h-[44px] pl-10 pr-16 bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl text-base sm:text-sm text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 focus:outline-none focus:border-indigo-500"
                />
                {searchQuery && (
                  <button
                    type="button"
                    onClick={() => setSearchQuery('')}
                    className="absolute right-2 top-1/2 -translate-y-1/2 min-h-[36px] px-2.5 text-xs font-medium text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100"
                  >
                    Clear
                  </button>
                )}
              </div>
            )}

            {/* Quick-Capture Bar (hidden in Search tab to keep search focused) */}
            {currentFilter !== 'search' && (
              <>
                {todaySpotlights.length > 0 && currentFilter !== 'today' && (
                  <section
                    aria-label="Today's Reminders"
                    className="rounded-2xl border border-indigo-200/80 dark:border-indigo-900/60 bg-indigo-50/50 dark:bg-indigo-950/25 p-3.5 space-y-2"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="inline-flex h-2 w-2 rounded-full bg-indigo-500 animate-pulse" />
                        <h2 className="text-xs font-semibold uppercase tracking-wider text-indigo-700 dark:text-indigo-300">
                          Today&apos;s Reminders ({todaySpotlights.length})
                        </h2>
                      </div>
                      <button
                        type="button"
                        onClick={() => setCurrentFilter('today')}
                        className="text-[11px] font-medium text-indigo-600 dark:text-indigo-400 hover:underline"
                      >
                        See all
                      </button>
                    </div>
                    <div className="grid gap-1.5 sm:grid-cols-2">
                      {todaySpotlights.slice(0, 4).map((item) => (
                        <button
                          key={item.id}
                          type="button"
                          onClick={() => setActiveReminder(item)}
                          className="flex items-center justify-between gap-2 px-3 py-2 rounded-xl bg-white/90 dark:bg-zinc-900/90 border border-indigo-100 dark:border-indigo-900/50 text-left hover:border-indigo-400 transition-colors"
                        >
                          <span className="text-xs font-medium text-zinc-900 dark:text-zinc-100 truncate">
                            {item.title}
                          </span>
                          {item.checklist.length > 0 && (
                            <span className="text-[10px] font-mono text-zinc-500 dark:text-zinc-400 shrink-0">
                              {item.checklist.filter((c) => c.completed).length}/{item.checklist.length}
                            </span>
                          )}
                        </button>
                      ))}
                    </div>
                  </section>
                )}
              <div
                className={`rounded-2xl border transition-colors ${
                  isDragOver
                    ? 'border-indigo-500 bg-indigo-50/40 dark:bg-indigo-950/20'
                    : 'border-zinc-200 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-900/50'
                } p-2 flex flex-col sm:flex-row items-stretch sm:items-center gap-2`}
              >
                <input
                  type="text"
                  value={quickInput}
                  onChange={(e) => setQuickInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleQuickCapture(false);
                    }
                  }}
                  placeholder="Quick add a reminder or drop a note image anywhere..."
                  className="flex-1 min-h-[40px] px-3 bg-transparent text-base sm:text-sm text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 dark:placeholder-zinc-500 focus:outline-none"
                />
                <div className="flex items-center gap-1.5 shrink-0 justify-end">
                  {quickInput.trim() ? (
                    <>
                      <button
                        type="button"
                        disabled={quickParsing}
                        onClick={() => handleQuickCapture(true)}
                        className="min-h-[40px] px-3.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-xs font-medium rounded-xl transition-colors inline-flex items-center gap-1.5 whitespace-nowrap"
                      >
                        {quickParsing ? (
                          <>
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                            <span>Parsing...</span>
                          </>
                        ) : (
                          <span>AI Parse</span>
                        )}
                      </button>
                      <button
                        type="button"
                        onClick={() => handleQuickCapture(false)}
                        className="min-h-[40px] px-3.5 bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 text-xs font-medium rounded-xl hover:opacity-90 transition-opacity whitespace-nowrap"
                      >
                        Add
                      </button>
                    </>
                  ) : (
                    <button
                      type="button"
                      onClick={() => setCurrentFilter('scan')}
                      className="min-h-[40px] px-3 text-xs font-medium text-zinc-600 dark:text-zinc-300 hover:text-zinc-900 dark:hover:text-white hover:bg-zinc-200/60 dark:hover:bg-zinc-800 rounded-xl transition-colors inline-flex items-center gap-1.5 whitespace-nowrap"
                    >
                      <Camera className="w-3.5 h-3.5" />
                      <span>Open OCR Studio</span>
                    </button>
                  )}
                </div>
              </div>
              </>
            )}

            {/* Workspace Toolbar: Count + View Switcher + Sort */}
            <div className="flex items-center justify-between gap-4 pb-1">
              <div className="text-xs text-zinc-500 dark:text-zinc-400 font-mono-nums">
                {sortedReminders.length} {sortedReminders.length === 1 ? 'reminder' : 'reminders'}
              </div>

              <div className="flex items-center gap-2">
                {/* Segmented View Switcher */}
                <div className="flex items-center gap-0.5 p-0.5 bg-zinc-100 dark:bg-zinc-900 border border-zinc-200/80 dark:border-zinc-800 rounded-xl">
                  {(
                    [
                      { id: 'list', label: 'List', icon: <List className="w-3.5 h-3.5" /> },
                      { id: 'grid', label: 'Grid', icon: <LayoutGrid className="w-3.5 h-3.5" /> },
                      { id: 'board', label: 'Board', icon: <Columns3 className="w-3.5 h-3.5" /> },
                    ] as const
                  ).map((mode) => {
                    const active = viewMode === mode.id;
                    return (
                      <button
                        key={mode.id}
                        type="button"
                        onClick={() => setViewMode(mode.id)}
                        className={`min-h-[36px] px-2.5 py-1 rounded-lg text-xs font-medium inline-flex items-center gap-1.5 transition-colors whitespace-nowrap ${
                          active
                            ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-sm'
                            : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200'
                        }`}
                      >
                        {mode.icon}
                        <span className="hidden sm:inline">{mode.label}</span>
                      </button>
                    );
                  })}
                </div>

                {/* Sort Select */}
                <div className="flex items-center gap-1.5 min-h-[38px] px-2.5 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl text-xs text-zinc-600 dark:text-zinc-300">
                  <ArrowUpDown className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as 'dueDate' | 'priority' | 'title')}
                    className="bg-transparent font-medium focus:outline-none cursor-pointer text-xs"
                    aria-label="Sort reminders"
                  >
                    <option value="dueDate" className="bg-white dark:bg-zinc-900">Due Date</option>
                    <option value="priority" className="bg-white dark:bg-zinc-900">Priority</option>
                    <option value="title" className="bg-white dark:bg-zinc-900">Title</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Reminders Content */}
            {sortedReminders.length === 0 && viewMode !== 'board' ? (
              <div className="border border-zinc-200 dark:border-zinc-800 rounded-2xl p-10 text-center space-y-4">
                <div className="w-11 h-11 rounded-2xl bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-zinc-500 dark:text-zinc-400 flex items-center justify-center mx-auto">
                  <Camera className="w-5 h-5" />
                </div>
                <div className="max-w-sm mx-auto space-y-1">
                  <h3 className="font-display text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                    No reminders here
                  </h3>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">
                    {searchQuery
                      ? `No matches for "${searchQuery}".`
                      : 'Scan a handwritten note, bill, or screenshot in OCR Studio to create your next reminder.'}
                  </p>
                </div>
                <div className="flex items-center justify-center gap-2 pt-1">
                  <button
                    onClick={() => setCurrentFilter('scan')}
                    className="min-h-[44px] px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium rounded-xl inline-flex items-center gap-1.5 transition-colors"
                  >
                    <Camera className="w-3.5 h-3.5" />
                    <span>Open OCR Studio</span>
                  </button>
                  <button
                    onClick={() => setIsAddOpen(true)}
                    className="min-h-[44px] px-4 py-2 bg-white dark:bg-zinc-900 hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-700 dark:text-zinc-200 border border-zinc-200 dark:border-zinc-800 text-xs font-medium rounded-xl inline-flex items-center gap-1.5 transition-colors"
                  >
                    <Plus className="w-3.5 h-3.5" />
                    <span>New Reminder</span>
                  </button>
                </div>
              </div>
            ) : viewMode === 'list' ? (
              <div className="border border-zinc-200 dark:border-zinc-800 rounded-2xl overflow-hidden divide-y-0">
                {sortedReminders.map((reminder) => (
                  <ReminderCard
                    key={reminder.id}
                    reminder={reminder}
                    variant="row"
                    isSelected={activeReminder?.id === reminder.id}
                    onToggleComplete={handleToggleComplete}
                    onToggleStar={handleToggleStar}
                    onDelete={handleDeleteReminder}
                    onClick={() => setActiveReminder(reminder)}
                  />
                ))}
              </div>
            ) : viewMode === 'grid' ? (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {sortedReminders.map((reminder) => (
                  <ReminderCard
                    key={reminder.id}
                    reminder={reminder}
                    variant="card"
                    isSelected={activeReminder?.id === reminder.id}
                    onToggleComplete={handleToggleComplete}
                    onToggleStar={handleToggleStar}
                    onClick={() => setActiveReminder(reminder)}
                  />
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-start">
                {[
                  { title: 'Urgent & Overdue', items: urgentOrOverdue },
                  { title: 'Scheduled', items: scheduled },
                  { title: 'Completed', items: doneList },
                ].map((col) => (
                  <div
                    key={col.title}
                    className="bg-zinc-50/70 dark:bg-zinc-900/30 border border-zinc-200/80 dark:border-zinc-800/80 rounded-2xl p-3 space-y-2.5"
                  >
                    <div className="flex items-center justify-between px-1">
                      <h3 className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">
                        {col.title}
                      </h3>
                      <span className="font-mono-nums text-[11px] text-zinc-400">
                        {col.items.length}
                      </span>
                    </div>
                    <div className="space-y-2.5">
                      {col.items.length > 0 ? (
                        col.items.map((reminder) => (
                          <ReminderCard
                            key={reminder.id}
                            reminder={reminder}
                            variant="card"
                            isSelected={activeReminder?.id === reminder.id}
                            onToggleComplete={handleToggleComplete}
                            onToggleStar={handleToggleStar}
                            onClick={() => setActiveReminder(reminder)}
                          />
                        ))
                      ) : (
                        <div className="py-8 text-center text-xs text-zinc-400 dark:text-zinc-500">
                          Empty
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </main>
        )}
      </div>

      {/* Right Split-Pane (Desktop) / Bottom Sheet (Mobile) Inspector */}
      {activeReminder && (
        <ReminderInspector
          reminder={activeReminder}
          onClose={() => setActiveReminder(null)}
          onUpdate={handleUpdateReminder}
          onDelete={handleDeleteReminder}
          onSendPushReminder={handleSendPushForReminder}
          pushEnabled={pushEnabled}
        />
      )}

      {/* Mobile Fixed 4-Tab Bottom Navigation Bar */}
      {!hideMobileBottomNav && (
        <BottomNav
          activeTab={activeMobileTab}
          onSelectTab={handleMobileTabSelect}
          inboxCount={activeCount}
          todayCount={todayCount}
        />
      )}

      {isAddOpen && (
        <AddReminderModal
          onClose={() => setIsAddOpen(false)}
          onAdd={handleAddReminder}
        />
      )}

      {isAuthOpen && (
        <AuthModal
          onClose={() => setIsAuthOpen(false)}
          onAuthSuccess={() => {
            showToast('Signed in. Syncing reminders from Supabase.');
          }}
        />
      )}

      <CommandPalette
        isOpen={isCommandOpen}
        onClose={() => setIsCommandOpen(false)}
        reminders={reminders}
        onSelectReminder={(r) => {
          setCurrentFilter('all');
          setActiveReminder(r);
        }}
        onSelectView={(v) => setCurrentFilter(v)}
        onOpenScan={() => {
          setCurrentFilter('scan');
          setActiveReminder(null);
        }}
        onOpenAdd={() => setIsAddOpen(true)}
        isDark={isDark}
        onToggleTheme={() => setIsDark((d) => !d)}
      />

      {/* Subtle Toast Banner */}
      {toast && (
        <div className="fixed bottom-20 lg:bottom-5 right-4 left-4 sm:left-auto sm:right-5 z-50 sm:max-w-sm">
          <div
            className={`px-4 py-3 rounded-xl border shadow-lg text-xs font-medium flex items-center justify-between gap-3 ${
              toast.type === 'error'
                ? 'bg-rose-50 dark:bg-rose-950 border-rose-200 dark:border-rose-800 text-rose-800 dark:text-rose-200'
                : 'bg-zinc-900 dark:bg-zinc-100 border-zinc-800 dark:border-zinc-200 text-white dark:text-zinc-900'
            }`}
          >
            <span className="truncate">{toast.message}</span>
            <div className="flex items-center gap-2 shrink-0">
              {toast.actionLabel && toast.onAction && (
                <button
                  type="button"
                  onClick={() => {
                    toast.onAction?.();
                    setToast(null);
                  }}
                  className="font-semibold underline underline-offset-2 text-indigo-300 dark:text-indigo-600"
                >
                  {toast.actionLabel}
                </button>
              )}
              <button onClick={() => setToast(null)} className="opacity-70 hover:opacity-100 p-1">
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
