"""Combine real Snap2 corrections, synthetic cursive crops, and optional IAM/CVL data
into leakage-free train / val / test TSV manifests grouped by writer_id.
"""
from __future__ import annotations

import argparse
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class SampleRow:
    abs_path: Path
    label: str
    source: str
    writer_id: str


def read_manifest(manifest_path: Path) -> List[SampleRow]:
    if not manifest_path.is_file():
        return []
    rows: List[SampleRow] = []
    base = manifest_path.parent
    for idx, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines()):
        if idx == 0 and line.lower().startswith("rel_path\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        rel = parts[0].strip()
        label = parts[1].strip()
        source = parts[2].strip() if len(parts) > 2 else manifest_path.parent.name
        writer_id = parts[3].strip() if len(parts) > 3 else f"{source}_default"
        img_path = (base / rel).resolve()
        if label and img_path.is_file():
            rows.append(SampleRow(abs_path=img_path, label=label, source=source, writer_id=writer_id))
    return rows


def build_splits(
    source_manifests: List[Path],
    out_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, int]:
    all_rows: List[SampleRow] = []
    seen_keys = set()
    for m in source_manifests:
        for row in read_manifest(m):
            key = hashlib.sha256(f"{row.abs_path}|{row.label}".encode("utf-8")).hexdigest()
            if key not in seen_keys:
                seen_keys.add(key)
                all_rows.append(row)

    # Group by writer_id so the same writer/font/page never leaks across train & val/test
    by_writer: Dict[str, List[SampleRow]] = {}
    for row in all_rows:
        by_writer.setdefault(row.writer_id, []).append(row)

    rng = random.Random(seed)
    writers = list(by_writer.keys())
    rng.shuffle(writers)

    train_rows: List[SampleRow] = []
    val_rows: List[SampleRow] = []
    test_rows: List[SampleRow] = []

    if len(writers) < 3:
        # Fallback when very few writers exist: split rows directly
        shuffled = list(all_rows)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        train_rows = shuffled[:n_train]
        val_rows = shuffled[n_train : n_train + n_val]
        test_rows = shuffled[n_train + n_val :]
    else:
        total = len(all_rows)
        target_train = int(round(total * train_ratio))
        target_val = int(round(total * val_ratio))
        for w in writers:
            group = by_writer[w]
            if len(train_rows) < target_train:
                train_rows.extend(group)
            elif len(val_rows) < target_val:
                val_rows.extend(group)
            else:
                test_rows.extend(group)
        if not test_rows and len(val_rows) > 1:
            test_rows.append(val_rows.pop())

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for name, bucket in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        lines = ["rel_path\tlabel\tsource\twriter_id"]
        for r in bucket:
            try:
                rel = r.abs_path.relative_to(out_dir.resolve()).as_posix()
            except ValueError:
                rel = r.abs_path.as_posix()
            lines.append(f"{rel}\t{r.label}\t{r.source}\t{r.writer_id}")
        (out_dir / f"{name}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts[name] = len(bucket)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-free train/val/test OCR manifests.")
    parser.add_argument(
        "--manifests",
        nargs="*",
        default=[
            "training/dataset/real/labels.tsv",
            "training/dataset/synth/labels.tsv",
            "training/dataset/iam/labels.tsv",
            "training/dataset/cvl/labels.tsv",
        ],
        help="Input labels.tsv manifests to merge",
    )
    parser.add_argument("--out", type=str, default="training/dataset/splits", help="Output split directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    counts = build_splits([Path(p) for p in args.manifests], Path(args.out), seed=args.seed)
    print(f"Built splits in {args.out}: {counts}")


if __name__ == "__main__":
    main()
