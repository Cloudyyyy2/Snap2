# Snap2 Offline Cursive OCR Training Pipeline (`training/`)

This directory is **strictly isolated from `python_ocr/`** so Snap2's runtime stays 100% offline, PyTorch-free, and ONNX-only (`test_r4_offline_onnx_only_compliance`).

## Zero-Token Architecture
- **Local PC (CPU):** Automatically collects real user corrections from Snap2 (`training/dataset/real/`), generates realistic synthetic cursive training data (`training/synth_cursive.py`), builds train/val/test splits (`training/dataset_builder.py`), and benchmarks ONNX models (`training/eval_benchmark.py`).
- **Cloud GPU (Google Colab / Kaggle Free T4 — `0` LLM tokens):** Runs `training/train_parseq_colab.ipynb` to fine-tune PARSeq (`baudm/parseq`) and exports a single drop-in `model.onnx` for `public/models/parseq/model.onnx`.

---

## Quickstart Workflow

### 1. Generate Synthetic Cursive Data (Offline, Local CPU)
```powershell
.\.venv\Scripts\python.exe training\synth_cursive.py --count 2000 --out training\dataset\synth
```
*(Drop any extra cursive `.ttf` fonts into `training/fonts/` to expand handwriting style coverage.)*

### 2. Automatic Real Data Collection (Built into Snap2 UI)
Whenever you edit/correct a recognized OCR line in **Snap2 (`OcrStudio`)**, Snap2 automatically saves the cropped line PNG and your corrected label into `training/dataset/real/labels.tsv` (deduplicated by SHA-256 hash).

### 3. Build Leakage-Free Train / Val / Test Splits
```powershell
.\.venv\Scripts\python.exe training\dataset_builder.py --out training\dataset\splits
```

### 4. Measure Current Baseline CER / WER (Offline ONNX)
```powershell
.\.venv\Scripts\python.exe training\eval_benchmark.py --model public\models\parseq\model.onnx --manifest training\dataset\splits\test.tsv
```

### 5. Fine-Tune PARSeq on Free GPU & Deploy
1. Zip `training/dataset/` and upload to Google Drive (or Kaggle).
2. Open `training/train_parseq_colab.ipynb` in Google Colab with **T4 GPU** enabled and click **Run All**.
3. Download the exported `parseq_cursive.onnx` into `training/candidates/parseq_cursive.onnx`.
4. Benchmark it locally:
   ```powershell
   .\.venv\Scripts\python.exe training\eval_benchmark.py --model training\candidates\parseq_cursive.onnx --manifest training\dataset\splits\test.tsv
   ```
5. If CER/WER beats the baseline and `pytest python_ocr/tests -v` passes, replace `public/models/parseq/model.onnx`!
