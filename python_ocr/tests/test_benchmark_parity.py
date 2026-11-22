from __future__ import annotations

import io
import sys
import unicodedata
from pathlib import Path
from typing import List, Tuple

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from python_ocr.postprocess import (
    LEXICON_SET,
    build_ligature_variants,
    correct_grocery_context,
    correct_text_with_lexicon,
    correct_word_with_lexicon,
    repair_conservative_phrases,
)
from python_ocr.recognizer import get_engine, recognize_image_bytes
from python_ocr.server import app


def compute_levenshtein(a: str, b: str) -> int:
    al, bl = len(a), len(b)
    if al == 0:
        return bl
    if bl == 0:
        return al
    d = [[0] * (bl + 1) for _ in range(al + 1)]
    for i in range(al + 1):
        d[i][0] = i
    for j in range(bl + 1):
        d[0][j] = j
    for i in range(1, al + 1):
        for j in range(1, bl + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[al][bl]


def evaluate_character_accuracy(ground_truth: str, hypothesis: str) -> Tuple[float, float]:
    gt = unicodedata.normalize("NFC", ground_truth or "").strip()
    hyp = unicodedata.normalize("NFC", hypothesis or "").strip()
    if not gt and not hyp:
        return 0.0, 1.0
    if not gt:
        return 1.0, 0.0
    edit_dist = compute_levenshtein(gt, hyp)
    cer = edit_dist / len(gt)
    return cer, max(0.0, 1.0 - cer)


def _load_font(candidates: List[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _make_synthetic_png(
    lines: List[str],
    font_candidates: List[str],
    *,
    size: int = 34,
    ruled: bool = False,
) -> bytes:
    img = Image.new("RGB", (560, 180), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    if ruled:
        for y in (68, 128):
            draw.line([(0, y), (560, y)], fill=(205, 212, 224), width=1)
    font = _load_font(font_candidates, size)
    y = 24
    for line in lines:
        draw.text((28, y), line, fill=(24, 24, 27), font=font)
        y += 60
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize(
    ("raw_word", "expected"),
    [
        ("Sundned", "Sundried"),
        ("Pundned", "Sundried"),
        ("Virwilli", "Broccoli"),
        ("Virwlli", "Broccoli"),
        ("browlli", "broccoli"),
        ("tormw", "tomorrow"),
        ("egge", "eggs"),
        ("Solman", "Salmon"),
        ("chulat", "chocolat"),
        ("twhy", "turkey"),
        ("fr", "for"),
        ("fm", "for"),
        ("Jolnun", "Salmon"),
        ("Vioulli", "Broccoli"),
        ("Vjoulli", "Broccoli"),
        ("Viralli", "Broccoli"),
        ("timmav", "tomorrow"),
        ("spinod", "spinach"),
    ],
)
@pytest.mark.parametrize("capture_mode", ["auto", "grocery"])
def test_cursive_ligatures_and_misreads_parity(raw_word: str, expected: str, capture_mode: str) -> None:
    corrected_word, changed = correct_word_with_lexicon(raw_word)
    assert changed is True
    assert corrected_word == expected
    assert correct_text_with_lexicon(raw_word, capture_mode=capture_mode) == expected


def test_browlli_is_not_in_lexicon_set() -> None:
    assert "browlli" not in LEXICON_SET
    for alias in ("vioulli", "vjoulli", "viralli", "timmav", "spinod", "spinad", "jolnun"):
        assert alias not in LEXICON_SET
    assert "broccoli" in LEXICON_SET
    assert "tomatoes" in LEXICON_SET


def test_build_ligature_variants_deterministic() -> None:
    sundned_variants = build_ligature_variants("sundned")
    assert "sundried" in sundned_variants

    tormw_variants = build_ligature_variants("tormw")
    assert "tomorrow" in tormw_variants

    twhy_variants = build_ligature_variants("twhy")
    assert "turhy" in twhy_variants


@pytest.mark.parametrize("capture_mode", ["auto", "grocery"])
def test_conservative_phrases_parity(capture_mode: str) -> None:
    assert repair_conservative_phrases("as PEAS") == "asparagus"
    assert repair_conservative_phrases("As peas") == "Asparagus"
    assert repair_conservative_phrases("Pan an chocolat") == "Pain au chocolat"
    assert repair_conservative_phrases("Panan chrolat") == "Pain au chocolat"
    assert repair_conservative_phrases("Sundried torthel") == "Sundried tomatoes"
    assert repair_conservative_phrases("Sundried parshel") == "Sundried tomatoes"
    assert repair_conservative_phrases("lunch fir tomorrow") == "lunch for tomorrow"

    assert correct_text_with_lexicon("as PEAS", capture_mode=capture_mode) == "asparagus"
    assert correct_text_with_lexicon("As peas", capture_mode=capture_mode) == "Asparagus"
    assert correct_text_with_lexicon("Pan an chocolat", capture_mode=capture_mode) == "Pain au chocolat"
    assert correct_text_with_lexicon("Panan chrolat", capture_mode=capture_mode) == "Pain au chocolat"
    assert correct_text_with_lexicon("Pundried torthel", capture_mode=capture_mode) == "Sundried tomatoes"
    assert correct_text_with_lexicon("Pundried parshel", capture_mode=capture_mode) == "Sundried tomatoes"
    assert correct_text_with_lexicon("lunch fir tomorrow", capture_mode=capture_mode) == "lunch for tomorrow"
    assert (
        correct_text_with_lexicon("milk\neggs\nPan an chocolat", capture_mode=capture_mode)
        == "milk\neggs\nPain au chocolat"
    )
    assert correct_grocery_context("milk\neggs\nPain au chocolat") == "milk\neggs\nPain au chocolat"


@pytest.mark.parametrize("capture_mode", ["auto", "grocery"])
def test_proper_noun_and_valid_short_word_guards(capture_mode: str) -> None:
    proper_noun_line = "Dr. Smith met Alice Bramwell at 3pm on 5/12 for $12.50"
    assert correct_text_with_lexicon(proper_noun_line, capture_mode=capture_mode) == proper_noun_line
    assert correct_text_with_lexicon("call Dr. Alice at 3pm", capture_mode=capture_mode) == "call Dr. Alice at 3pm"

    short_words = "they this that then tray tiny"
    assert correct_text_with_lexicon(short_words, capture_mode=capture_mode) == short_words

    common_words = "boss client code gym mom ship"
    assert correct_text_with_lexicon(common_words, capture_mode=capture_mode) == common_words

    for word in (
        "tray", "buns", "pesto", "met", "busy", "army", "lady", "jury",
        "holy", "wary", "wavy", "bony", "edgy", "troy", "ably", "oily", "lazy", "bury",
    ):
        corrected, changed = correct_word_with_lexicon(word)
        assert changed is False
        assert corrected == word

    for num_tok in ("3pm", "10:30am", "$45.50", "2026"):
        corrected, changed = correct_word_with_lexicon(num_tok)
        assert changed is False
        assert corrected == num_tok


BENCHMARK_SAMPLES = [
    (
        "sample-01-cursive-groceries",
        "buy milk\nfresh eggs\nbread and cheese",
        "buy rnilk\nfnesh eggs\nbnead and cheese",
    ),
    (
        "sample-02-cursive-appointment",
        "doctor appointment tomorrow\ncall Dr. Alice at 3pm\ninvoice $45.50",
        "doctar appointnnent tommorow\ncall Dr. Alice at 3pm\ninvoice $45.50",
    ),
    (
        "sample-03-printed-bill",
        "Pay electric bill $114\nDue Friday 5pm\nAccount 8021",
        "Pay electnc bill $114\nDue Friday 5pm\nAccount 8o21",
    ),
    (
        "sample-04-cursive-proper-nouns",
        "Bramwell project review\nschedule meeting next week\npriority high",
        "Bramwell pnoject review\nschedule rneeting next week\npnionity high",
    ),
    (
        "sample-05-cursive-slanted-tasks",
        "morning workout at 7am\nprepare report for class\nfinish work today",
        "moming workout at 7am\npnepare report for dass\nfinish vvork today",
    ),
    (
        "sample-06-printed-schedule",
        "check meter reading\nsubmit report by Monday\nmeeting at 10:30am",
        "check meter readin9\nsubmit repont by Monday\nmeetin9 at 10:30am",
    ),
]


@pytest.mark.parametrize(("sample_id", "ground_truth", "raw_hypothesis"), BENCHMARK_SAMPLES)
def test_ts_benchmark_samples_character_accuracy(sample_id: str, ground_truth: str, raw_hypothesis: str) -> None:
    corrected = correct_text_with_lexicon(raw_hypothesis, capture_mode="auto")
    cer, accuracy = evaluate_character_accuracy(ground_truth, corrected)
    assert accuracy >= 0.90, f"{sample_id}: accuracy={accuracy:.3f}, corrected={corrected!r}"
    assert cer <= 0.10


def test_ts_benchmark_aggregate_character_accuracy() -> None:
    total_gt = 0
    total_ed = 0
    for _, gt, raw in BENCHMARK_SAMPLES:
        corrected = correct_text_with_lexicon(raw, capture_mode="auto")
        total_gt += len(unicodedata.normalize("NFC", gt).strip())
        total_ed += compute_levenshtein(
            unicodedata.normalize("NFC", gt).strip(),
            unicodedata.normalize("NFC", corrected).strip(),
        )
    overall_accuracy = 1.0 - (total_ed / total_gt)
    assert overall_accuracy >= 0.90


def test_e2e_recognize_image_bytes_printed_fixture() -> None:
    if not get_engine().is_available():
        pytest.skip("Offline ONNX TrOCR model files not present")

    gt = "buy milk tomorrow\ncall dentist at 3pm"
    png_bytes = _make_synthetic_png(
        gt.splitlines(),
        [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"],
        size=34,
        ruled=False,
    )
    res = recognize_image_bytes(png_bytes, ocr_profile="printed", export_stages=False)
    assert res["recognizer"] == "python-local-ocr"
    assert res["needsUserPurpose"] is False
    assert res["detectedStyle"] == "printed"
    assert "python-lexicon-postprocess" in res["appliedSteps"]
    _, accuracy = evaluate_character_accuracy(gt, res["text"])
    assert accuracy >= 0.90, f"Printed E2E accuracy={accuracy:.3f}, text={res['text']!r}"


def test_e2e_recognize_image_bytes_cursive_handwriting_fixture() -> None:
    if not get_engine().is_available():
        pytest.skip("Offline ONNX TrOCR model files not present")

    gt = "buy milk tomorrow\nbread and eggs"
    png_bytes = _make_synthetic_png(
        gt.splitlines(),
        [r"C:\Windows\Fonts\segoepr.ttf", r"C:\Windows\Fonts\comic.ttf", r"C:\Windows\Fonts\arial.ttf"],
        size=36,
        ruled=True,
    )
    res = recognize_image_bytes(png_bytes, ocr_profile="handwriting", export_stages=False)
    assert res["recognizer"] == "python-local-ocr"
    assert res["needsUserPurpose"] is False
    assert res["detectedStyle"] == "handwriting"
    assert "deslant-lines" in res["appliedSteps"]
    assert "python-lexicon-postprocess" in res["appliedSteps"]
    _, accuracy = evaluate_character_accuracy(gt, res["text"])
    assert accuracy >= 0.90, f"Handwriting E2E accuracy={accuracy:.3f}, text={res['text']!r}"


def test_r4_offline_onnx_only_compliance() -> None:
    for forbidden in ("tesseract", "pytesseract", "torch"):
        assert forbidden not in sys.modules

    py_dir = Path(__file__).resolve().parent.parent
    for py_file in py_dir.glob("*.py"):
        src = py_file.read_text(encoding="utf-8")
        assert "pytesseract" not in src
        assert "tesseract" not in src.lower()

    health = TestClient(app).get("/health")
    assert health.status_code == 200
    assert health.json()["offlineOnly"] is True
