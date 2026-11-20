export type Category = 'Bills' | 'Events' | 'Tasks' | 'Travel' | 'Work' | 'Personal' | 'Health';
export type Priority = 'high' | 'medium' | 'low';
export type OcrProfile = 'auto' | 'printed' | 'handwriting';
export type DetectedStyle = 'printed' | 'handwriting' | 'mixed';
export type CaptureMode = 'auto' | 'grocery' | 'receipt' | 'recipe' | 'fridge';

export interface OcrNormalizedBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface OcrLineResult {
  text: string;
  confidence: number;
  bbox: { x0: number; y0: number; x1: number; y1: number };
  normBox?: OcrNormalizedBox;
}

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
  completionImage?: string; // optional photo proof when marked complete
  exportDirName?: string; // host ocr-exports/run-* folder name
  ocrText?: string;
  ocrEngine?: string;
  completed: boolean;
  starred: boolean;
  createdAt: string;
}
