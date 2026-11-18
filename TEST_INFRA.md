# E2E Test Infra: Snap2 OCR Pipeline

## Test Philosophy
- Opaque-box, requirement-driven testing across Offline Local and Vision AI OCR pipelines.
- Multi-tier coverage: Category-Partition (Tier 1), Boundary Value Analysis (Tier 2), Cross-Feature (Tier 3), and Real-World Application Workloads (Tier 4).
- Character Error Rate (`CER = Levenshtein(GT, Hyp) / GT.length`) benchmark threshold: Character accuracy `(1 - CER) >= 90%`.

## Feature Inventory
| # | Feature | Source (requirement) | Tier 1 | Tier 2 | Tier 3 |
|---|---------|---------------------|:------:|:------:|:------:|
| 1 | Grayscale-Preserving Normalization | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ |
| 2 | Cursive Slant Normalization | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ |
| 3 | Ligature-Safe Line Segmentation | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ |
| 4 | TrOCR Beam Search Decoding | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ |
| 5 | Contextual Post-Processing | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ |
| 6 | OCR Studio UI Output & Mode Selection | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ |
| 7 | Recognizer Test Fix & Stability | ORIGINAL_REQUEST §R5 | 5 | 5 | ✓ |
| 8 | Character Accuracy Benchmark (>= 90%) | ORIGINAL_REQUEST §R5 | 5 | 5 | ✓ |

## Test Architecture
- Test runner: `npm run test:unit` (`vitest run`)
- Type checking: `npm run lint` (`tsc --noEmit`)
- Test files:
  - `tests/unit/ocrBenchmark.test.ts` — Automated cursive and printed accuracy benchmark (Tier 1-4)
  - `tests/unit/handwritingRecognizer.test.ts` — Recognizer pipeline & `saveGrayscalePng` memory stability
  - `tests/unit/wordDataset.test.ts` — Bidirectional confusion rules & proper noun protection
  - `tests/unit/geometry.test.ts` — Deslanting shear & slant angle estimation
  - `tests/unit/handwritingCleanup.test.ts` — Ligature-safe line segmentation & ascender/descender preservation

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Multi-line Cursive Shopping List | F1, F2, F3, F4, F5, F8 | High |
| 2 | Handwritten Appointment Reminder with Proper Nouns & Times | F1, F3, F4, F5, F8 | High |
| 3 | Printed Document with Numbers and Currency | F1, F5, F8 | Medium |
| 4 | Offline Local OCR Full Pipeline Execution | F1, F2, F3, F4, F6, F7, F8 | High |
| 5 | Strongly Slanted Cursive with Interlocking Ascenders/Descenders | F1, F2, F3, F4, F5, F8 | High |
