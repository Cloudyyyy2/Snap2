export type Category = 'Bills' | 'Events' | 'Tasks' | 'Travel' | 'Work' | 'Personal' | 'Health';
export type Priority = 'high' | 'medium' | 'low';
export type OcrProfile = 'auto' | 'printed' | 'handwriting';
export type DetectedStyle = 'printed' | 'handwriting' | 'mixed';

export interface ChecklistItem {
  id: string;
  text: string;
  completed: boolean;
}

export interface Reminder {
  id: string;
  title: string;
  description: string;
  dueDate: string; // ISO string
  category: Category;
  priority: Priority;
  checklist: ChecklistItem[];
  tags: string[];
  image?: string; // base64 data url or image url
  ocrText?: string;
  ocrEngine?: string;
  completed: boolean;
  starred: boolean;
  createdAt: string;
}
