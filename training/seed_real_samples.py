"""Seed verified real handwritten notebook line crops into training/dataset/real/.

Extracts clean line crops using Snap2's preprocessing pipeline and appends hash-deduplicated
ground-truth rows into training/dataset/real/labels.tsv.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import List, Tuple

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python_ocr.preprocess import run_preprocessing

SEED_SOURCES: List[Tuple[Path, str, List[str]]] = [
    (
        Path(
            r"C:\Users\paste\.gemini\antigravity\brain\a39037d9-3bf5-4972-8be5-9007b443c493\.user_uploaded\media_1795289813276_c93e39f5.jpg"
        ),
        "real_notebook_grocery_01",
        [
            "bread",
            "milk",
            "crisps",
            "eggs",
            "pain au chocolat",
            "lunch for tomorrow",
            "salmon",
            "asparagus",
            "bacon",
            "turkey",
            "spinach",
            "turkey mince",
            "sundried tomatoes",
            "broccoli",
        ],
    ),
]


def seed_real_samples(real_dir: Path | None = None) -> int:
    target_dir = real_dir or (REPO_ROOT / "training" / "dataset" / "real")
    images_dir = target_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / "labels.tsv"

    if not manifest_path.is_file():
        manifest_path.write_text("rel_path\tlabel\tsource\twriter_id\toriginal_text\n", encoding="utf-8")

    added = 0
    for img_path, writer_id, labels in SEED_SOURCES:
        if not img_path.is_file():
            continue
        prep = run_preprocessing(img_path.read_bytes(), ocr_profile="auto", export_stages=False)
        if len(prep.line_boxes) != len(labels):
            print(
                f"[seed_real_samples] Skipping {img_path.name}: expected {len(labels)} lines, got {len(prep.line_boxes)}"
            )
            continue

        for box, label in zip(prep.line_boxes, labels):
            crop = prep.parseq_gray[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1]
            if crop.size == 0:
                continue
            padded = cv2.copyMakeBorder(crop, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=255)
            ok, encoded = cv2.imencode(".png", padded)
            if not ok:
                continue
            raw_bytes = encoded.tobytes()
            h = hashlib.sha256(raw_bytes + b"|" + label.encode("utf-8")).hexdigest()[:12]
            fname = f"real_{h}.png"
            dst = images_dir / fname
            if dst.is_file():
                continue
            dst.write_bytes(raw_bytes)
            with manifest_path.open("a", encoding="utf-8") as fh:
                fh.write(f"images/{fname}\t{label}\tsnap2_real_seed\t{writer_id}\t\n")
            added += 1

    print(f"Seeded {added} new real handwritten line crops -> {manifest_path}")
    return added


if __name__ == "__main__":
    seed_real_samples()
