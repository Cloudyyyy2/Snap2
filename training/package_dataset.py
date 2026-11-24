"""Package Snap2's train/val/test OCR splits and referenced PNGs into a portable Colab-ready zip.

Rewrites manifest paths to be cleanly relative to splits/ inside the archive so Google Colab
loads 100% of images without broken local paths.
"""
from __future__ import annotations

import argparse
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def package_dataset(splits_dir: Path, out_zip: Path) -> Dict[str, object]:
    splits_dir = splits_dir.resolve()
    dataset_root = splits_dir.parent.resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    archived_images: Dict[Path, str] = {}
    rewritten_manifests: Dict[str, List[str]] = {}
    split_counts: Dict[str, int] = {}
    source_counts: Counter[str] = Counter()

    for split_name in ("train", "val", "test"):
        tsv_path = splits_dir / f"{split_name}.tsv"
        if not tsv_path.is_file():
            continue
        lines_out = ["rel_path\tlabel\tsource\twriter_id"]
        count = 0
        for idx, line in enumerate(tsv_path.read_text(encoding="utf-8").splitlines()):
            if idx == 0 and line.lower().startswith("rel_path\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            raw_rel = parts[0].strip()
            label = parts[1].strip()
            source = parts[2].strip() if len(parts) > 2 else "unknown"
            writer_id = parts[3].strip() if len(parts) > 3 else "default"

            img_abs = Path(raw_rel)
            if not img_abs.is_absolute():
                img_abs = (splits_dir / raw_rel).resolve()
            if not img_abs.is_file() or not label:
                continue

            if img_abs not in archived_images:
                try:
                    arc_name = img_abs.relative_to(dataset_root).as_posix()
                except ValueError:
                    arc_name = f"extra/{img_abs.name}"
                archived_images[img_abs] = arc_name
            else:
                arc_name = archived_images[img_abs]

            # Path from splits/<split>.tsv to <arc_name> inside the zip
            rel_from_splits = f"../{arc_name}"
            lines_out.append(f"{rel_from_splits}\t{label}\t{source}\t{writer_id}")
            count += 1
            source_counts[source] += 1

        rewritten_manifests[f"splits/{split_name}.tsv"] = lines_out
        split_counts[split_name] = count

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for arc_manifest, lines in rewritten_manifests.items():
            zf.writestr(arc_manifest, "\n".join(lines) + "\n")
        for img_abs, arc_name in archived_images.items():
            zf.write(img_abs, arcname=arc_name)

    size_mb = out_zip.stat().st_size / (1024.0 * 1024.0)
    return {
        "zip_path": str(out_zip),
        "size_mb": round(size_mb, 2),
        "unique_images": len(archived_images),
        "splits": split_counts,
        "sources": dict(sorted(source_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Snap2 OCR dataset into a portable Colab zip.")
    parser.add_argument("--splits", type=str, default="training/dataset/splits", help="Splits directory")
    parser.add_argument("--out", type=str, default="training/dataset.zip", help="Output zip file path")
    args = parser.parse_args()

    summary = package_dataset(Path(args.splits), Path(args.out))
    print(
        f"Packaged {summary['unique_images']} images -> {summary['zip_path']} "
        f"({summary['size_mb']:.2f} MB) | Splits: {summary['splits']}"
    )


if __name__ == "__main__":
    main()
