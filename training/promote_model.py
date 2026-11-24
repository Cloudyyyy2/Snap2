"""Safe local ONNX model promotion gate for Snap2 PARSeq models.

Compares a candidate .onnx model against the active public/models/parseq/model.onnx on the
held-out test split, backs up the active model to model.onnx.bak, runs pytest verification,
auto-rolls back on any test failure, and logs promotion history to training/model_history.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.eval_benchmark import evaluate_model


def promote_model(
    candidate_path: Path,
    manifest_path: Path,
    active_model_path: Path,
    limit: int = 150,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate ONNX model not found: {candidate_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Test manifest not found: {manifest_path}")

    print(f"Benchmarking baseline:  {active_model_path}")
    base_res = evaluate_model(active_model_path, manifest_path, limit=limit)
    print(f"Benchmarking candidate: {candidate_path}")
    cand_res = evaluate_model(candidate_path, manifest_path, limit=limit)

    print("\n=== PARSeq ONNX Promotion Comparison ===")
    print(
        f"Baseline  | CER: {base_res['cer']*100:6.2f}% | WER: {base_res['wer']*100:6.2f}% | "
        f"Exact: {base_res['exact_accuracy']*100:6.2f}% | Latency: {base_res['avg_latency_ms']:.1f} ms"
    )
    print(
        f"Candidate | CER: {cand_res['cer']*100:6.2f}% | WER: {cand_res['wer']*100:6.2f}% | "
        f"Exact: {cand_res['exact_accuracy']*100:6.2f}% | Latency: {cand_res['avg_latency_ms']:.1f} ms"
    )

    improved = cand_res["cer"] < base_res["cer"] or (
        cand_res["cer"] == base_res["cer"] and cand_res["exact_accuracy"] >= base_res["exact_accuracy"]
    )
    if dry_run:
        print(f"\n[DRY RUN] Would promote: {improved} (no files modified).")
        return improved

    if not improved and not force:
        print("\n[BLOCKED] Candidate CER did not beat baseline. Re-run with --force to override.")
        return False

    backup_path = active_model_path.with_suffix(".onnx.bak")
    shutil.copy2(active_model_path, backup_path)
    shutil.copy2(candidate_path, active_model_path)
    print(f"\nBacked up active model -> {backup_path}")
    print(f"Installed candidate -> {active_model_path}")

    # Run full Python OCR pytest suite as final safety gate
    print("\nRunning pytest python_ocr/tests -v ...")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "python_ocr/tests", "-v"],
        cwd=str(REPO_ROOT),
        check=False,
    )
    if proc.returncode != 0:
        print("\n[ROLLBACK] Pytest failed! Restoring previous model.onnx from backup.")
        shutil.copy2(backup_path, active_model_path)
        return False

    # Record promotion in training/model_history.json
    history_path = REPO_ROOT / "training" / "model_history.json"
    history = []
    if history_path.is_file():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            history = []
    history.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "candidate": str(candidate_path),
            "baseline_cer": base_res["cer"],
            "candidate_cer": cand_res["cer"],
            "baseline_exact": base_res["exact_accuracy"],
            "candidate_exact": cand_res["exact_accuracy"],
            "samples_evaluated": cand_res["samples"],
        }
    )
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"\n[PROMOTED] Model verified and logged to {history_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely benchmark and promote a fine-tuned PARSeq ONNX model.")
    parser.add_argument("--candidate", type=str, required=True, help="Path to candidate .onnx file")
    parser.add_argument("--manifest", type=str, default="training/dataset/splits/test.tsv")
    parser.add_argument("--active", type=str, default="public/models/parseq/model.onnx")
    parser.add_argument("--limit", type=int, default=150, help="Max test samples to benchmark (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Benchmark only without modifying files")
    parser.add_argument("--force", action="store_true", help="Promote even if CER did not improve")
    args = parser.parse_args()

    ok = promote_model(
        candidate_path=Path(args.candidate),
        manifest_path=Path(args.manifest),
        active_model_path=Path(args.active),
        limit=args.limit,
        dry_run=args.dry_run,
        force=args.force,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
