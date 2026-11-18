export type ReminderCategory =
  | 'Bills'
  | 'Events'
  | 'Tasks'
  | 'Travel'
  | 'Work'
  | 'Personal'
  | 'Health';

export interface ParsedReminderPayload {
  title: string;
  description: string;
  dueDate: string;
  category: ReminderCategory;
  priority: 'high' | 'medium' | 'low';
  checklist: Array<{ id: string; text: string; completed: boolean }>;
  tags: string[];
}

function inferCategory(text: string): ReminderCategory {
  const lower = text.toLowerCase();
  if (/\b(bill|invoice|rent|electric|water|utility|payment|pay|tax|due|credit card|insurance)\b/.test(lower)) {
    return 'Bills';
  }
  if (/\b(doctor|dentist|clinic|med|rx|prescription|appointment|checkup|therapy|health|gym)\b/.test(lower)) {
    return 'Health';
  }
  if (/\b(flight|hotel|train|airport|boarding|ticket|trip|itinerary|passport|travel)\b/.test(lower)) {
    return 'Travel';
  }
  if (/\b(meeting|standup|sync|deadline|report|client|office|project|deck|review)\b/.test(lower)) {
    return 'Work';
  }
  if (/\b(party|dinner|birthday|concert|wedding|event|rsvp|meetup)\b/.test(lower)) {
    return 'Events';
  }
  if (/\b(mom|dad|family|gift|home|call)\b/.test(lower)) {
    return 'Personal';
  }
  return 'Tasks';
}

function inferPriority(text: string): 'high' | 'medium' | 'low' {
  const lower = text.toLowerCase();
  if (/\b(urgent|asap|immediately|important|critical|overdue|must|today)\b/.test(lower)) {
    return 'high';
  }
  if (/\b(someday|whenever|low priority|optional|later)\b/.test(lower)) {
    return 'low';
  }
  return 'medium';
}

function inferDueDate(text: string): string {
  const now = new Date();
  const base = new Date(now);
  base.setHours(9, 0, 0, 0);

  const lower = text.toLowerCase();
  let dateMatched = false;

  if (/\btoday\b/.test(lower)) {
    dateMatched = true;
  } else if (/\btomorrow\b/.test(lower)) {
    base.setDate(base.getDate() + 1);
    dateMatched = true;
  } else {
    // Numeric date like 5/12 or 12/25/2026
    const slashMatch = text.match(/\b(\d{1,2})\/(\d{1,2})(?:\/(\d{2,4}))?\b/);
    if (slashMatch) {
      const m = parseInt(slashMatch[1], 10);
      const d = parseInt(slashMatch[2], 10);
      let y = slashMatch[3] ? parseInt(slashMatch[3], 10) : now.getFullYear();
      if (y < 100) y += 2000;
      if (m >= 1 && m <= 12 && d >= 1 && d <= 31) {
        base.setFullYear(y, m - 1, d);
        if (!slashMatch[3] && base.getTime() < now.getTime() - 86400000) {
          base.setFullYear(y + 1);
        }
        dateMatched = true;
      }
    }
  }

  // Time like 3pm, 3:30 pm, 14:30
  const time12 = text.match(/\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b/i);
  const time24 = !time12 ? text.match(/\b([01]?\d|2[0-3]):([0-5]\d)\b/) : null;

  if (time12) {
    let hours = parseInt(time12[1], 10);
    const mins = time12[2] ? parseInt(time12[2], 10) : 0;
    const ampm = time12[3].toLowerCase();
    if (ampm === 'pm' && hours < 12) hours += 12;
    if (ampm === 'am' && hours === 12) hours = 0;
    base.setHours(hours, mins, 0, 0);
    if (!dateMatched && base.getTime() <= now.getTime()) {
      base.setDate(base.getDate() + 1);
    }
    dateMatched = true;
  } else if (time24) {
    const hours = parseInt(time24[1], 10);
    const mins = parseInt(time24[2], 10);
    base.setHours(hours, mins, 0, 0);
    if (!dateMatched && base.getTime() <= now.getTime()) {
      base.setDate(base.getDate() + 1);
    }
    dateMatched = true;
  }

  if (!dateMatched) {
    base.setDate(base.getDate() + 1);
  }

  return base.toISOString();
}

/**
 * Deterministic offline parser that converts OCR text into a structured Reminder
 * without requiring cloud API connectivity.
 */
export function parseReminderLocally(
  rawText: string,
  extraPrompt?: string,
  extraTags: string[] = []
): ParsedReminderPayload {
  const lines = rawText
    .split(/\r?\n/)
    .map((l) => l.replace(/^[-*•\u2022]\s*/, '').replace(/^\[[ xX]?\]\s*/, '').trim())
    .filter(Boolean);

  const combined = [rawText, extraPrompt || ''].filter(Boolean).join('\n');
  const firstLine = lines[0] || extraPrompt?.trim() || 'Handwritten Note';
  const title = firstLine.length > 72 ? `${firstLine.slice(0, 69).trim()}...` : firstLine;

  const description =
    lines.length > 1
      ? lines.slice(1).join('\n')
      : rawText.trim() || extraPrompt?.trim() || 'Captured from note';

  const checklist =
    lines.length > 1
      ? lines.slice(0, 50).map((line, i) => ({
          id: `chk-${i + 1}`,
          text: line,
          completed: false,
        }))
      : [];

  const category = inferCategory(combined);
  const priority = inferPriority(combined);
  const dueDate = inferDueDate(combined);

  const tags = Array.from(new Set([...extraTags, category.toLowerCase()]));

  return {
    title,
    description,
    dueDate,
    category,
    priority,
    checklist,
    tags,
  };
}
