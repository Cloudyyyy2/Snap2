import { Category, ChecklistItem, Priority, Reminder } from '../types';
import { getSupabaseClient } from '../lib/supabase';

const STORAGE_KEY = 'snapremind_reminders_v1';
const META_STORAGE_KEY = 'snapremind_meta_v1';

const tomorrow = new Date(Date.now() + 86400000).toISOString();
const nextWeek = new Date(Date.now() + 7 * 86400000).toISOString();
const pastDue = new Date(Date.now() - 3600000).toISOString();

const VALID_CATEGORIES: Category[] = ['Bills', 'Events', 'Tasks', 'Travel', 'Work', 'Personal', 'Health'];

export const INITIAL_REMINDERS: Reminder[] = [
  {
    id: 'rem-1',
    title: 'Electricity Bill Due',
    description: 'City Power & Light Statement #4892-B. Amount due: $114.50. Avoid late fees.',
    dueDate: tomorrow,
    category: 'Bills',
    priority: 'high',
    checklist: [
      { id: 'c1', text: 'Log into portal', completed: true },
      { id: 'c2', text: 'Verify meter reading', completed: false },
      { id: 'c3', text: 'Submit autopay or credit card', completed: false }
    ],
    tags: ['utilities', 'bill', 'urgent'],
    completed: false,
    starred: true,
    createdAt: new Date().toISOString()
  },
  {
    id: 'rem-2',
    title: 'Flight to San Francisco (UA 1420)',
    description: 'Confirmation code: Z9X7QR. Terminal 2, Gate B4. Boarding starts at 08:30 AM.',
    dueDate: nextWeek,
    category: 'Travel',
    priority: 'medium',
    checklist: [
      { id: 'c1', text: 'Check in online 24h prior', completed: false },
      { id: 'c2', text: 'Pack passport & charger', completed: false },
      { id: 'c3', text: 'Book airport ride', completed: false }
    ],
    tags: ['flight', 'trip', 'united'],
    completed: false,
    starred: true,
    createdAt: new Date().toISOString()
  },
  {
    id: 'rem-3',
    title: 'Dentist Checkup & Cleaning',
    description: 'Dr. Emily Vance, Smiles Dental Clinic, Suite 400. Bring insurance card.',
    dueDate: nextWeek,
    category: 'Health',
    priority: 'medium',
    checklist: [
      { id: 'c1', text: 'Confirm appointment SMS', completed: true },
      { id: 'c2', text: 'Brush & floss before visit', completed: false }
    ],
    tags: ['health', 'doctor'],
    completed: false,
    starred: false,
    createdAt: new Date().toISOString()
  },
  {
    id: 'rem-4',
    title: 'Q3 Product Roadmap Review',
    description: 'Present metrics and milestones to leadership team in Conference Room B.',
    dueDate: pastDue,
    category: 'Work',
    priority: 'high',
    checklist: [
      { id: 'c1', text: 'Finalize slide deck', completed: true },
      { id: 'c2', text: 'Send calendar invite with agenda', completed: true }
    ],
    tags: ['work', 'meeting', 'review'],
    completed: true,
    starred: false,
    createdAt: new Date(Date.now() - 86400000 * 2).toISOString()
  }
];

interface ReminderMeta {
  priority?: Priority;
  checklist?: ChecklistItem[];
  tags?: string[];
  starred?: boolean;
  image?: string;
  ocrText?: string;
  ocrEngine?: string;
}

function getMetaMap(): Record<string, ReminderMeta> {
  try {
    const raw = localStorage.getItem(META_STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveReminderMeta(reminder: Reminder): void {
  try {
    const map = getMetaMap();
    map[reminder.id] = {
      priority: reminder.priority,
      checklist: reminder.checklist,
      tags: reminder.tags,
      starred: reminder.starred,
      image: reminder.image,
      ocrText: reminder.ocrText,
      ocrEngine: reminder.ocrEngine,
    };
    localStorage.setItem(META_STORAGE_KEY, JSON.stringify(map));
  } catch (err) {
    console.warn('Could not persist reminder metadata locally:', err);
  }
}

export function getStoredReminders(): Reminder[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(INITIAL_REMINDERS));
      return INITIAL_REMINDERS;
    }
    return JSON.parse(raw);
  } catch (err) {
    console.error('Failed to load reminders from localStorage', err);
    return INITIAL_REMINDERS;
  }
}

export function saveStoredReminders(reminders: Reminder[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(reminders));
  } catch (err) {
    console.error('Failed to save reminders to localStorage', err);
  }
}

function normalizeCategory(cat: string | null | undefined): Category {
  if (!cat) return 'Tasks';
  const matched = VALID_CATEGORIES.find((c) => c.toLowerCase() === cat.toLowerCase());
  return matched || 'Tasks';
}

export async function fetchSupabaseReminders(userId: string): Promise<Reminder[]> {
  const supabase = await getSupabaseClient();
  if (!supabase) return getStoredReminders();

  const { data, error } = await supabase
    .from('reminders')
    .select('id, user_id, title, description, remind_at, category, sent, image_path, created_at, updated_at')
    .eq('user_id', userId)
    .order('remind_at', { ascending: true });

  if (error) throw error;

  const metaMap = getMetaMap();

  return (data || []).map((row: any) => {
    const meta = metaMap[row.id] || {};
    return {
      id: row.id,
      title: row.title || 'Untitled Reminder',
      description: row.description || '',
      dueDate: row.remind_at || row.created_at || new Date().toISOString(),
      category: normalizeCategory(row.category),
      priority: meta.priority || 'medium',
      checklist: meta.checklist || [],
      tags: meta.tags || ['cloud'],
      image: row.image_path || meta.image,
      ocrText: meta.ocrText,
      ocrEngine: meta.ocrEngine,
      completed: Boolean(row.sent),
      starred: Boolean(meta.starred),
      createdAt: row.created_at || new Date().toISOString(),
    };
  });
}

export async function createSupabaseReminder(userId: string, reminder: Reminder): Promise<Reminder> {
  const supabase = await getSupabaseClient();
  if (!supabase) return reminder;

  let imagePath: string | null = null;
  if (reminder.image && reminder.image.startsWith('data:')) {
    try {
      const matches = reminder.image.match(/^data:(.+);base64,(.+)$/);
      if (matches) {
        const ext = matches[1].split('/')[1] || 'png';
        const fileName = `${userId}/${Date.now()}.${ext}`;
        const byteChars = atob(matches[2]);
        const byteNums = new Array(byteChars.length);
        for (let i = 0; i < byteChars.length; i++) {
          byteNums[i] = byteChars.charCodeAt(i);
        }
        const blob = new Blob([new Uint8Array(byteNums)], { type: matches[1] });
        const { error: uploadErr } = await supabase.storage
          .from('screenshots')
          .upload(fileName, blob, { upsert: true });
        if (!uploadErr) {
          const { data: urlData } = supabase.storage.from('screenshots').getPublicUrl(fileName);
          imagePath = urlData?.publicUrl || fileName;
        }
      }
    } catch (uploadErr) {
      console.warn('Screenshot upload skipped:', uploadErr);
    }
  } else if (reminder.image) {
    imagePath = reminder.image;
  }

  const { data, error } = await supabase
    .from('reminders')
    .insert({
      user_id: userId,
      title: reminder.title,
      description: reminder.description,
      remind_at: reminder.dueDate,
      category: reminder.category,
      sent: reminder.completed,
      image_path: imagePath,
    })
    .select()
    .single();

  if (error) throw error;

  const saved: Reminder = {
    ...reminder,
    id: data.id,
    image: imagePath || reminder.image,
    createdAt: data.created_at || reminder.createdAt,
  };
  saveReminderMeta(saved);
  return saved;
}

export async function updateSupabaseReminder(userId: string, reminder: Reminder): Promise<void> {
  saveReminderMeta(reminder);
  const supabase = await getSupabaseClient();
  if (!supabase) return;

  const { error } = await supabase
    .from('reminders')
    .update({
      title: reminder.title,
      description: reminder.description,
      remind_at: reminder.dueDate,
      category: reminder.category,
      sent: reminder.completed,
      updated_at: new Date().toISOString(),
    })
    .eq('id', reminder.id)
    .eq('user_id', userId);

  if (error) throw error;
}

export async function deleteSupabaseReminder(userId: string, id: string): Promise<void> {
  const supabase = await getSupabaseClient();
  if (!supabase) return;

  const { error } = await supabase
    .from('reminders')
    .delete()
    .eq('id', id)
    .eq('user_id', userId);

  if (error) throw error;
}
