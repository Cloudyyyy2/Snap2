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


def load_vocabulary() -> List[str]:
    words: List[str] = list(EXTRA_GROCERY_PHRASES)
    lex_path = REPO_ROOT / "python_ocr" / "compact_lexicon.json"
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


def render_cursive_sample(text: str, font_path: Path, rng: random.Random) -> np.ndarray:
    font_size = rng.randint(28, 44)
    font = ImageFont.truetype(str(font_path), font_size)

    # Measure text generously
    dummy = Image.new("L", (16, 16), 255)
    draw_d = ImageDraw.Draw(dummy)
    bbox = draw_d.textbbox((0, 0), text, font=font)
    tw = max(16, bbox[2] - bbox[0])
    th = max(16, bbox[3] - bbox[1])

    pad_x = rng.randint(14, 28)
    pad_y = rng.randint(10, 20)
    w = tw + pad_x * 2
    h = th + pad_y * 2

    paper_val = rng.randint(228, 252)
    ink_val = rng.randint(12, 85)
    img = Image.new("L", (w, h), paper_val)
    draw = ImageDraw.Draw(img)

    # Optional faint ruled notebook line
    if rng.random() < 0.65:
        rule_y = int(h * rng.uniform(0.68, 0.86))
        rule_val = rng.randint(180, 215)
        draw.line([(0, rule_y), (w - 1, rule_y)], fill=rule_val, width=1)

    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, fill=ink_val, font=font)
    arr = np.array(img, dtype=np.uint8)

    # Slight slant/shear simulating cursive lean
    shear = rng.uniform(-0.18, 0.12)
    angle = rng.uniform(-3.0, 3.0)
    m_rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    m_rot[0, 1] += shear
    arr = cv2.warpAffine(arr, m_rot, (w, h), flags=cv2.INTER_LINEAR, borderValue=paper_val)

    # Elastic stroke warp
    if rng.random() < 0.75:
        arr = _apply_elastic_warp(arr, rng)

    # Uneven phone-camera lighting gradient
    if rng.random() < 0.6:
        grad_x = np.linspace(rng.uniform(-18, 18), rng.uniform(-18, 18), w, dtype=np.float32)
        arr = np.clip(arr.astype(np.float32) + grad_x[None, :], 0, 255).astype(np.uint8)

    # Slight camera defocus / sensor noise
    if rng.random() < 0.45:
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

        img = render_cursive_sample(text, font_path, rng)
        fname = f"synth_{idx:06d}.png"
        cv2.imwrite(str(img_dir / fname), img)
        writer_id = f"font_{font_path.stem.lower()}"
        rows.append(f"images/{fname}\t{text}\tsynth_cursive\t{writer_id}")

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
