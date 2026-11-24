"""Offline synthetic cursive word/line dataset generator for Snap2 PARSeq fine-tuning.

Uses installed cursive/handwriting TrueType fonts (plus any custom .ttf/.otf in training/fonts/)
and applies realistic notebook degradations (slant, elastic stroke warp, ruled lines, uneven
illumination) before optionally matching Snap2's preprocessing distribution.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent

# Priority Windows cursive / handwriting fonts
DEFAULT_CURSIVE_FONT_PATTERNS = (
    "segoepr.ttf",
    "segoeprb.ttf",
    "segoesc.ttf",
    "segoescb.ttf",
    "Inkfree.ttf",
    "BRADHITC.TTF",
    "BRUSHSCI.TTF",
    "FRSCRIPT.TTF",
    "Gabriola.ttf",
    "MISTRAL.TTF",
    "PRISTINA.TTF",
    "SCRIPTBL.TTF",
    "VIVALDII.TTF",
    "mvboli.ttf",
    "comic.ttf",
)

EXTRA_GROCERY_PHRASES = (
    "pain au chocolat",
    "lunch for tomorrow",
    "turkey mince",
    "sundried tomatoes",
    "olive oil",
    "greek yogurt",
    "brown bread",
    "whole milk",
    "free range eggs",
    "cheddar cheese",
    "fresh spinach",
    "cherry tomatoes",
    "red onion",
    "garlic cloves",
    "chicken breast",
    "smoked salmon",
    "green asparagus",
    "streaky bacon",
    "broccoli florets",
    "pita chips",
    "dryer sheets",
    "parmesan shred",
)


def discover_cursive_fonts(extra_dir: Path | None = None) -> List[Path]:
    fonts: List[Path] = []
    if extra_dir and extra_dir.is_dir():
        for ext in ("*.ttf", "*.otf"):
            fonts.extend(sorted(extra_dir.glob(ext)))

    win_fonts = Path(r"C:\Windows\Fonts")
    if win_fonts.is_dir():
        for fname in DEFAULT_CURSIVE_FONT_PATTERNS:
            p = win_fonts / fname
            if p.is_file():
                fonts.append(p)

    # Linux/macOS fallback scan
    for sys_dir in (Path("/usr/share/fonts"), Path("/Library/Fonts")):
        if sys_dir.is_dir() and not fonts:
            fonts.extend(sorted(sys_dir.rglob("*.ttf"))[:12])

    return fonts


ISSUE_PROFILES: Tuple[str, ...] = (
    "clean_cursive",
    "ruled_notebook",
    "heavy_slant",
    "faint_ink",
    "thick_bleed",
    "shadow_gradient",
    "motion_blur",
    "broken_strokes",
    "adjacent_bleed",
    "spiral_edge",
)


def load_vocabulary() -> List[str]:
    words: List[str] = list(EXTRA_GROCERY_PHRASES)
    for fname in ("grocery_lexicon.json", "compact_lexicon.json"):
        lex_path = REPO_ROOT / "python_ocr" / fname
        if lex_path.is_file():
            try:
                data = json.loads(lex_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    words.extend(str(w).strip() for w in data if isinstance(w, str) and len(w.strip()) >= 2)
            except Exception:
                pass
    # Deduplicate preserving order
    return list(dict.fromkeys(words))


def _apply_elastic_warp(gray: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = gray.shape
    if h < 16 or w < 24:
        return gray
    alpha = rng.uniform(1.2, 3.2)
    sigma = rng.uniform(5.0, 9.0)
    dx = cv2.GaussianBlur(
        np.random.default_rng(rng.randint(0, 2**31 - 1)).uniform(-1.0, 1.0, (h, w)).astype(np.float32),
        (0, 0),
        sigma,
    ) * alpha
    dy = cv2.GaussianBlur(
        np.random.default_rng(rng.randint(0, 2**31 - 1)).uniform(-1.0, 1.0, (h, w)).astype(np.float32),
        (0, 0),
        sigma,
    ) * (alpha * 0.7)
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(
        gray,
        grid_x + dx,
        grid_y + dy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def render_cursive_sample(text: str, font_path: Path, rng: random.Random, issue: str = "clean_cursive") -> np.ndarray:
    font_size = rng.randint(28, 44)
    font = ImageFont.truetype(str(font_path), font_size)

    # Measure text generously
    dummy = Image.new("L", (16, 16), 255)
    draw_d = ImageDraw.Draw(dummy)
    bbox = draw_d.textbbox((0, 0), text, font=font)
    tw = max(16, bbox[2] - bbox[0])
    th = max(16, bbox[3] - bbox[1])

    pad_x = rng.randint(18, 32) if issue == "spiral_edge" else rng.randint(14, 28)
    pad_y = rng.randint(10, 20)
    w = tw + pad_x * 2
    h = th + pad_y * 2

    paper_val = rng.randint(228, 252)
    ink_val = rng.randint(115, 165) if issue == "faint_ink" else rng.randint(12, 80)
    img = Image.new("L", (w, h), paper_val)
    draw = ImageDraw.Draw(img)

    # Ruled notebook line
    if issue == "ruled_notebook" or (issue != "clean_cursive" and rng.random() < 0.45):
        rule_y = int(h * rng.uniform(0.65, 0.86))
        rule_val = rng.randint(165, 208)
        draw.line([(0, rule_y), (w - 1, rule_y)], fill=rule_val, width=1)

    # Simulated left-margin spiral coil shadows
    if issue == "spiral_edge":
        for sy in range(4, h - 4, max(12, h // 3)):
            draw.line([(0, sy), (rng.randint(8, 16), sy + rng.randint(-2, 2))], fill=rng.randint(45, 105), width=2)

    # Simulated top/bottom ascender/descender bleed from adjacent lines
    if issue == "adjacent_bleed":
        for _ in range(rng.randint(1, 3)):
            bx = rng.randint(pad_x, max(pad_x + 4, w - pad_x))
            if rng.random() < 0.5:
                draw.line([(bx, 0), (bx + rng.randint(-4, 4), rng.randint(6, 12))], fill=ink_val + 20, width=2)
            else:
                draw.line([(bx, h - 1), (bx + rng.randint(-4, 4), h - rng.randint(6, 12))], fill=ink_val + 20, width=2)

    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, fill=ink_val, font=font)
    arr = np.array(img, dtype=np.uint8)

    # Thick marker / heavy ink bleed
    if issue == "thick_bleed":
        arr = cv2.erode(arr, np.ones((2, 2), dtype=np.uint8), iterations=1)

    # Slant / shear & rotation
    if issue == "heavy_slant":
        shear = rng.choice([rng.uniform(-0.32, -0.18), rng.uniform(0.14, 0.24)])
        angle = rng.uniform(-5.5, 5.5)
    else:
        shear = rng.uniform(-0.16, 0.10)
        angle = rng.uniform(-3.0, 3.0)
    m_rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    m_rot[0, 1] += shear
    arr = cv2.warpAffine(arr, m_rot, (w, h), flags=cv2.INTER_LINEAR, borderValue=paper_val)

    # Elastic stroke warp
    if rng.random() < 0.75:
        arr = _apply_elastic_warp(arr, rng)

    # Broken cursive strokes / pen skips
    if issue == "broken_strokes":
        for _ in range(rng.randint(2, 5)):
            rx = rng.randint(pad_x, max(pad_x + 2, w - pad_x))
            ry = rng.randint(pad_y, max(pad_y + 2, h - pad_y))
            arr[max(0, ry - 1) : min(h, ry + 2), max(0, rx - 1) : min(w, rx + 2)] = paper_val

    # Shadow gradient (harsh uneven phone lighting)
    if issue == "shadow_gradient":
        grad_x = np.linspace(rng.uniform(-55, -20), rng.uniform(10, 35), w, dtype=np.float32)
        if rng.random() < 0.5:
            grad_x = grad_x[::-1]
        arr = np.clip(arr.astype(np.float32) + grad_x[None, :], 0, 255).astype(np.uint8)
    elif rng.random() < 0.45:
        grad_x = np.linspace(rng.uniform(-16, 16), rng.uniform(-16, 16), w, dtype=np.float32)
        arr = np.clip(arr.astype(np.float32) + grad_x[None, :], 0, 255).astype(np.uint8)

    # Motion blur / camera shake
    if issue == "motion_blur":
        ksize = rng.choice([3, 5])
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[ksize // 2, :] = 1.0 / ksize
        arr = cv2.filter2D(arr, -1, kernel)
    elif rng.random() < 0.4:
        arr = cv2.GaussianBlur(arr, (3, 3), rng.uniform(0.4, 0.9))

    if rng.random() < 0.5:
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, rng.uniform(2.0, 6.0), arr.shape)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return arr


def generate_dataset(out_dir: Path, count: int, seed: int = 42, fonts_dir: Path | None = None) -> Tuple[int, Path]:
    rng = random.Random(seed)
    fonts = discover_cursive_fonts(fonts_dir)
    if not fonts:
        raise RuntimeError("No TrueType fonts found. Place cursive .ttf files in training/fonts/.")

    vocab = load_vocabulary()
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: List[str] = ["rel_path\tlabel\tsource\twriter_id"]
    for idx in range(count):
        font_path = rng.choice(fonts)
        issue = ISSUE_PROFILES[idx % len(ISSUE_PROFILES)]
        # 75% single words (PARSeq word crops), 25% 2-word phrases
        if rng.random() < 0.75:
            text = rng.choice(vocab)
        else:
            text = f"{rng.choice(vocab)} {rng.choice(vocab)}"
            if len(text) > 22:
                text = rng.choice(vocab)

        # Randomize realistic casing (lowercase, Capitalized, ALL CAPS)
        r_case = rng.random()
        if r_case < 0.55:
            text = text.lower()
        elif r_case < 0.92:
            text = text[0].upper() + text[1:].lower()
        else:
            text = text.upper()

        img = render_cursive_sample(text, font_path, rng, issue=issue)
        fname = f"synth_{idx:06d}.png"
        cv2.imwrite(str(img_dir / fname), img)
        writer_id = f"font_{font_path.stem.lower()}"
        rows.append(f"images/{fname}\t{text}\tsynth_{issue}\t{writer_id}")

    manifest_path = out_dir / "labels.tsv"
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return count, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline synthetic cursive OCR training samples.")
    parser.add_argument("--count", type=int, default=1000, help="Number of synthetic cursive samples to generate")
    parser.add_argument("--out", type=str, default="training/dataset/synth", help="Output directory")
    parser.add_argument("--fonts-dir", type=str, default="training/fonts", help="Optional extra TTF fonts folder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    n, manifest = generate_dataset(
        out_dir=Path(args.out),
        count=args.count,
        seed=args.seed,
        fonts_dir=Path(args.fonts_dir),
    )
    print(f"Generated {n} synthetic cursive samples -> {manifest}")


if __name__ == "__main__":
    main()
