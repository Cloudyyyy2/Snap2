# Project: Snap2 OCR Strengthening & Cursive Support

## Architecture
- **Frontend & Preprocessing**: `src/components/OcrStudio.tsx`, `src/services/imageProcessor.ts`
- **Geometry & Preprocessing**: `src/lib/ocr/geometry.ts`, `src/lib/ocr/preprocess.ts`, `src/lib/ocr/handwriting/lines.ts`
- **Server Recognizer & Contextual Post-Processing**: `server/handwritingRecognizer.ts`, `src/lib/ocr/wordDataset.ts`, `src/lib/ocr/postprocess.ts`
- **Testing & Verification**: `tests/unit/handwritingRecognizer.test.ts`, `tests/unit/ocrBenchmark.test.ts`, `tests/unit/geometry.test.ts`, `tests/unit/wordDataset.test.ts`

## Code Layout
- `src/services/imageProcessor.ts` — Grayscale-preserving normalization in `preprocessForOCR`
- `src/components/OcrStudio.tsx` — OCR mode selector (Auto, Cursive/Handwriting, Printed, Offline Local), accuracy %, line-by-line confidence scores, detected style badge, copy text button
- `src/lib/ocr/geometry.ts` — Cursive slant estimation and shear deslanting transformation without character clipping
- `src/lib/ocr/handwriting/lines.ts` — Ligature-safe line segmentation with ascender/descender margins (0.32 line pitch) and ligature-aware chunk splitting
- `server/handwritingRecognizer.ts` — TrOCR beam search decoding (`num_beams: 4`, `max_new_tokens: 64`, `early_stopping: true`), deslanting integration, and `saveGrayscalePng` dimension fix
- `src/lib/ocr/wordDataset.ts` — Bidirectional cursive confusion rules (`rn`<->`m`, `cl`<->`d`, `vv`<->`w`, `ri`<->`n`), weighted edit distance, proper noun & numeric protection
- `tests/unit/ocrBenchmark.test.ts` — Lightweight cursive and printed benchmark dataset with ground-truth transcripts asserting `>= 90%` character accuracy
- `tests/unit/handwritingRecognizer.test.ts` — Unit test ensuring `saveGrayscalePng` passes after `auto-bbox-shrink`
- `vite.config.ts` — TypeScript type fix for `allowedHosts: true as const`

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Grayscale-Preserving Normalization | Stop hard adaptive binarization in `src/services/imageProcessor.ts` and `src/components/OcrStudio.tsx`; preserve anti-aliased continuous stroke gradations. | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Cursive Slant Normalization | Implement deslanting shear transformation and projection variance maximization in `src/lib/ocr/geometry.ts`. | M1 | ORIGINAL_REQUEST §R2 |
| 3 | Ligature-Safe Line Segmentation | Implement line segmentation with generous vertical ascender/descender margins and ligature-safe word splitting in `src/lib/ocr/handwriting/lines.ts`. | M1 | ORIGINAL_REQUEST §R2 |
| 4 | TrOCR Beam Search Decoding | Upgrade TrOCR decoding in `server/handwritingRecognizer.ts` to `num_beams: 4`, `max_new_tokens: 64`, `early_stopping: true`. | M1 | ORIGINAL_REQUEST §R3 |
| 5 | Contextual Post-Processing | Bidirectional cursive confusion rules (`rn`<->`m`, `cl`<->`d`, `vv`<->`w`, `ri`<->`n`), weighted edit distance, proper noun protection in `src/lib/ocr/wordDataset.ts`. | M1 | ORIGINAL_REQUEST §R3 |
| 6 | OCR Studio UI Output & Mode Selection | Mode selector (Auto, Cursive/Handwriting, Printed, Offline Local), accuracy %, line-by-line confidence scores, detected style badge, working copy text button in `src/components/OcrStudio.tsx`. | M1 | ORIGINAL_REQUEST §R4 |
| 7 | Recognizer Test Fix | Fix `VipsImage: memory area too small` error in `saveGrayscalePng` after `auto-bbox-shrink` in `server/handwritingRecognizer.ts`. | M1 | ORIGINAL_REQUEST §R5 |
| 8 | Verification Benchmark & Dataset | Lightweight cursive and printed test dataset with ground-truth transcripts in `tests/unit/ocrBenchmark.test.ts` achieving >= 90% character accuracy across Offline Local and Vision AI modes. | M1 | ORIGINAL_REQUEST §R5 |
| 9 | Zero Unit Test / Lint Errors | `npm run test:unit` and `npm run lint` pass with zero errors. | M1 | ORIGINAL_REQUEST §AC |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Full Pipeline Strengthening & Benchmark | All features F1 through F9 | none | DONE (68/68 unit tests passing, lint 0 errors, character accuracy 99.1% with CER 0.009) |

## Interface Contracts
### `src/lib/ocr/geometry.ts`
- `estimateCursiveSlantAngle(binary: Uint8ClampedArray, w: number, h: number, options?: DeslantOptions): number`
- `shearGrayscale(gray: Uint8ClampedArray, w: number, h: number, angleDeg: number, backgroundFill?: number): { data: Uint8ClampedArray; width: number; height: number; affineMatrix: number[] }`
- `deslantGrayscale(gray: Uint8ClampedArray, binary: Uint8ClampedArray, w: number, h: number, options?: DeslantOptions): DeslantResult`

### `src/lib/ocr/handwriting/lines.ts`
- `segmentLines(binary: Uint8ClampedArray, w: number, h: number): LineBox[]` (includes generous margin padding `max(12, round(H_line * 0.32))`)
- `splitWideLineBox(binary: Uint8ClampedArray, w: number, h: number, box: LineBox, maxAspect?: number): LineBox[]` (preserves unbroken cursive ligatures)

### `server/handwritingRecognizer.ts`
- `recognizeLines(lineCrops: CroppedLine[]): Promise<Array<{ text: string; confidence: number }>>` (with beam search: `num_beams: 4`, `max_new_tokens: 64`, `early_stopping: true`)
- `saveGrayscalePng(filePath: string, gray: Uint8ClampedArray, width: number, height: number): Promise<void>` (called with matching buffer dimensions `w, h`)

### `src/lib/ocr/wordDataset.ts`
- `OCR_CONFUSION_RULES`: Bidirectional rules for `rn`<->`m`, `cl`<->`d`, `vv`<->`w`, `ri`<->`n`, `ol`<->`d`, `li`<->`h`.
- `correctWordWithLexicon`: Cursive confusion penalty = 0.35, bypass capitalized proper nouns, numbers, currency, timestamps.
