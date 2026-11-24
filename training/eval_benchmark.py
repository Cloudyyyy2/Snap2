"""Offline CPU ONNX benchmark evaluator for Snap2 PARSeq models.

Computes Character Error Rate (CER), Word Error Rate (WER), exact-match accuracy,
and average ms/sample latency on any labels.tsv manifest.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python_ocr.postprocess import damerau_levenshtein
from python_ocr.recognizer import ParseqOcrEngine


def _edit_distance(a: str, b: str) -> int:
    return damerau_levenshtein(a, b)


def evaluate_model(model_path: Path, manifest_path: Path, limit: int = 0) -> dict:
    engine = ParseqOcrEngine(model_dir=model_path.parent)
    engine.model_path = model_path
    sidecar_cfg = model_path.parent / f"{model_path.stem}_config.json"
    if sidecar_cfg.is_file():
        engine.config_path = sidecar_cfg
    if not engine.is_available():
        raise FileNotFoundError(
            f"ONNX model or config not found: model={engine.model_path} config={engine.config_path}"
        )

    base = manifest_path.parent
    samples: List[Tuple[Path, str]] = []
    for idx, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines()):
        if idx == 0 and line.lower().startswith("rel_path\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        img_p = Path(parts[0].strip())
        if not img_p.is_absolute():
            img_p = (base / img_p).resolve()
        label = parts[1].strip()
        if label and img_p.is_file():
            samples.append((img_p, label))

    if limit > 0:
        samples = samples[:limit]

    total_chars = 0
    total_char_edits = 0
    total_words = 0
    total_word_edits = 0
    exact_matches = 0
    latencies_ms: List[float] = []
    worst: List[Tuple[int, str, str]] = []

    for img_path, gt in samples:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        t0 = time.perf_counter()
        pred, conf = engine.recognize_word(gray)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        gt_l = gt.strip().lower()
        pred_l = pred.strip().lower()
        c_edits = _edit_distance(pred_l, gt_l)
        total_char_edits += c_edits
        total_chars += max(1, len(gt_l))

        gt_words = gt_l.split()
        pred_words = pred_l.split()
        w_edits = abs(len(gt_words) - len(pred_words)) + sum(
            1 for gw, pw in zip(gt_words, pred_words) if gw != pw
        )
        total_word_edits += w_edits
        total_words += max(1, len(gt_words))

        if pred_l == gt_l:
            exact_matches += 1
        else:
            worst.append((c_edits, gt, pred))

    n = len(latencies_ms)
    cer = (total_char_edits / total_chars) if total_chars else 0.0
    wer = (total_word_edits / total_words) if total_words else 0.0
    exact_acc = (exact_matches / n) if n else 0.0
    avg_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    worst.sort(key=lambda x: x[0], reverse=True)
    return {
        "samples": n,
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "exact_accuracy": round(exact_acc, 4),
        "avg_latency_ms": round(avg_ms, 2),
        "top_errors": worst[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an ONNX PARSeq model offline.")
    parser.add_argument("--model", type=str, default="public/models/parseq/model.onnx")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Optional max samples to evaluate (0=all)")
    args = parser.parse_args()

    res = evaluate_model(Path(args.model), Path(args.manifest), limit=args.limit)
    print(
        f"Samples: {res['samples']} | CER: {res['cer']*100:.2f}% | "
        f"WER: {res['wer']*100:.2f}% | Exact: {res['exact_accuracy']*100:.2f}% | "
        f"Latency: {res['avg_latency_ms']:.1f} ms/crop"
    )
    if res["top_errors"]:
        print("Top errors (edits | ground_truth -> prediction):")
        for edits, gt, pred in res["top_errors"][:5]:
            print(f"  [{edits}] {gt!r} -> {pred!r}")


if __name__ == "__main__":
    main()
