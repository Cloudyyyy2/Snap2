import React from 'react';
import { Menu, Search, Camera } from 'lucide-react';
import { Category } from '../types';

interface NavbarProps {
  currentFilter: string;
  selectedCategory: Category | 'All';
  activeCount: number;
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  onOpenCommandPalette: () => void;
  onOpenUpload: () => void;
  onOpenMobileMenu: () => void;
}

const FILTER_LABELS: Record<string, string> = {
  all: 'Inbox',
  today: 'Today',
  upcoming: 'Upcoming',
  starred: 'Starred',
  completed: 'Completed',
};

export const Navbar: React.FC<NavbarProps> = ({
  currentFilter,
  selectedCategory,
  activeCount,
  searchQuery,
  setSearchQuery,
  onOpenCommandPalette,
  onOpenUpload,
  onOpenMobileMenu,
}) => {
  const viewTitle = FILTER_LABELS[currentFilter] || 'Inbox';

  return (
    <header className="sticky top-0 z-20 h-14 bg-white/90 dark:bg-zinc-950/90 backdrop-blur-md border-b border-zinc-200 dark:border-zinc-800/80 px-4 lg:px-6 flex items-center justify-between gap-4">
      {/* Left Zone: Mobile Menu + Breadcrumb */}
      <div className="flex items-center gap-3 min-w-0 shrink-0">
        <button
          onClick={onOpenMobileMenu}
          className="p-2 -ml-1 rounded-lg text-zinc-600 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-900 lg:hidden"
          aria-label="Open navigation"
        >
          <Menu className="w-4 h-4" />
        </button>

        <div className="flex items-center gap-2 text-sm whitespace-nowrap">
          <span className="font-display font-semibold text-zinc-900 dark:text-zinc-100">
            {viewTitle}
          </span>
          {selectedCategory !== 'All' && (
            <>
              <span className="text-zinc-400 dark:text-zinc-600">/</span>
              <span className="font-medium text-zinc-600 dark:text-zinc-300">
                {selectedCategory}
              </span>
            </>
          )}
          <span className="text-zinc-300 dark:text-zinc-700 hidden sm:inline">·</span>
          <span className="font-mono-nums text-xs text-zinc-500 dark:text-zinc-400 hidden sm:inline">
            {activeCount} open
          </span>
        </div>
      </div>

      {/* Center Zone: Clean Filter + Cmd+K Trigger */}
      <div className="flex-1 max-w-md hidden sm:flex items-center">
        <div className="relative w-full">
          <Search className="w-3.5 h-3.5 text-zinc-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Filter reminders..."
            className="w-full h-8 pl-8 pr-16 bg-zinc-100/80 dark:bg-zinc-900 border border-zinc-200/80 dark:border-zinc-800 rounded-lg text-xs text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 dark:placeholder-zinc-500 focus:outline-none focus:bg-white dark:focus:bg-zinc-950 focus:border-indigo-500 transition-colors"
          />
          {searchQuery ? (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-[11px] text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200 px-1.5 py-0.5"
            >
              Clear
            </button>
          ) : (
            <button
              type="button"
              onClick={onOpenCommandPalette}
              className="absolute right-1.5 top-1/2 -translate-y-1/2 font-mono-nums text-[10px] text-zinc-400 dark:text-zinc-500 bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 px-1.5 py-0.5 rounded hover:text-zinc-700 dark:hover:text-zinc-300"
              title="Open command palette"
            >
              ⌘K
            </button>
          )}
        </div>
      </div>

      {/* Right Zone: Single Primary Action */}
      <div className="flex items-center gap-2 shrink-0">
        <button
          onClick={onOpenCommandPalette}
          className="p-2 rounded-lg text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 hover:bg-zinc-100 dark:hover:bg-zinc-900 sm:hidden"
          aria-label="Search reminders"
        >
          <Search className="w-4 h-4" />
        </button>

        <button
          onClick={onOpenUpload}
          className="px-3.5 py-1.5 bg-zinc-900 hover:bg-zinc-800 dark:bg-zinc-100 dark:hover:bg-white text-white dark:text-zinc-900 text-xs font-medium rounded-lg transition-colors flex items-center gap-1.5 whitespace-nowrap shrink-0"
        >
          <Camera className="w-3.5 h-3.5" />
          <span>AI Scan</span>
        </button>
      </div>
    </header>
  );
};
