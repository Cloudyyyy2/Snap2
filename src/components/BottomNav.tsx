import React from 'react';
import { Camera, Inbox, Calendar, Search } from 'lucide-react';

export type MobileTab = 'scan' | 'inbox' | 'today' | 'search';

interface BottomNavProps {
  activeTab: MobileTab;
  onSelectTab: (tab: MobileTab) => void;
  inboxCount: number;
  todayCount: number;
}

export const BottomNav: React.FC<BottomNavProps> = ({
  activeTab,
  onSelectTab,
  inboxCount,
  todayCount,
}) => {
  const tabs: Array<{
    id: MobileTab;
    label: string;
    icon: React.ReactNode;
    count?: number;
  }> = [
    {
      id: 'scan',
      label: 'Scan',
      icon: <Camera className="w-[21px] h-[21px]" />,
    },
    {
      id: 'inbox',
      label: 'Inbox',
      icon: <Inbox className="w-[21px] h-[21px]" />,
      count: inboxCount,
    },
    {
      id: 'today',
      label: 'Today',
      icon: <Calendar className="w-[21px] h-[21px]" />,
      count: todayCount,
    },
    {
      id: 'search',
      label: 'Search',
      icon: <Search className="w-[21px] h-[21px]" />,
    },
  ];

  return (
    <nav
      aria-label="Primary mobile navigation"
      className="fixed bottom-0 left-0 right-0 z-30 lg:hidden bg-white/95 dark:bg-zinc-950/95 backdrop-blur-md border-t border-zinc-200 dark:border-zinc-800/80 pb-[env(safe-area-inset-bottom)]"
    >
      <div className="grid grid-cols-4 h-[60px] max-w-lg mx-auto px-1">
        {tabs.map((tab) => {
          const active = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => onSelectTab(tab.id)}
              className={`min-h-[44px] flex flex-col items-center justify-center gap-0.5 rounded-xl transition-colors active:scale-[0.97] ${
                active
                  ? 'text-indigo-600 dark:text-indigo-400'
                  : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
              }`}
              aria-current={active ? 'page' : undefined}
            >
              <div className="relative flex items-center justify-center">
                {tab.icon}
                {typeof tab.count === 'number' && tab.count > 0 && (
                  <span className="font-mono-nums absolute -top-1 -right-3 text-[10px] font-semibold leading-none text-zinc-700 dark:text-zinc-300">
                    {tab.count > 99 ? '99+' : tab.count}
                  </span>
                )}
              </div>
              <span className="text-[11px] font-medium tracking-tight whitespace-nowrap">
                {tab.label}
              </span>
            </button>
          );
        })}
      </div>
    </nav>
  );
};
