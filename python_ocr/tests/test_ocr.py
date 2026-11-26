from __future__ import annotations

import base64
import io
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from python_ocr.postprocess import (
    correct_grocery_context,
    correct_text_with_lexicon,
    detect_and_clean_hallucination,
    score_text_plausibility,
)
from python_ocr.preprocess import (
    LineBox,
    binarize_adaptive,
    deslant_grayscale,
    enhance_contrast,
    remove_ruled_lines,
    run_preprocessing,
    strip_line_edge_bleeds,
    strip_spiral_and_corner_artifacts,
)
from python_ocr.recognizer import (
    _accept_trocr_candidate,
    get_engine,
    get_parseq_engine,
    recognize_image_bytes,
)
from python_ocr.server import app


def _make_blank_png_bytes() -> bytes:
    img = Image.new("RGB", (320, 200), (245, 245, 245))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_grocery_png_bytes() -> bytes:
    img = Image.new("RGB", (460, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line([(0, 68), (460, 68)], fill=(200, 208, 220), width=1)
    draw.text((24, 34), "buy milk", fill=(20, 20, 20))
    draw.text((24, 100), "call dentist", fill=(20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_blank_image_flags_needs_user_purpose(tmp_path: Path):
    res = recognize_image_bytes(
        _make_blank_png_bytes(),
        ocr_profile="auto",
        export_stages=True,
        export_root_dir=str(tmp_path),
    )
    assert res["recognizer"] == "python-local-ocr"
    assert res["needsUserPurpose"] is True
    assert res["text"] == ""
    assert res["lines"] == []
    export_dir = Path(res["exportDir"])
    assert (export_dir / "01_original.png").is_file()
    assert (export_dir / "05_adaptive_binary.png").is_file()
    assert (export_dir / "extracted_text.txt").is_file()


def test_preprocessing_exports_all_named_stages(tmp_path: Path):
    prep = run_preprocessing(
        _make_grocery_png_bytes(),
        ocr_profile="handwriting",
        export_stages=True,
        export_root_dir=str(tmp_path),
    )
    assert prep.export_dir is not None
    run_dir = Path(prep.export_dir)
    assert run_dir.name.endswith("-handwriting")
    for name in [
        "01_original.png",
        "02_deskewed.png",
        "03b_background_flattened.png",
        "03c_ruled_lines_removed.png",
        "04_contrast_enhanced.png",
        "05_adaptive_binary.png",
    ]:
        assert (run_dir / name).is_file()
    assert len(prep.line_boxes) >= 1


def test_fastapi_health_and_ocr_offline(tmp_path: Path):
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["engine"] == "python-local-ocr"
    assert body["offlineOnly"] is True

    if not get_engine().is_available():
        return

    b64 = base64.b64encode(_make_grocery_png_bytes()).decode("ascii")
    resp = client.post(
        "/ocr",
        json={
            "image": f"data:image/png;base64,{b64}",
            "ocrProfile": "auto",
            "exportStages": True,
            "exportRootDir": str(tmp_path),
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["recognizer"] == "python-local-ocr"
    assert len(data["lines"]) >= 1
    assert 0.0 <= data["lines"][0]["normBox"]["x"] <= 1.0
    assert data["text"].strip() != ""
    assert (Path(data["exportDir"]) / "07_ocr_text_overlay.png").is_file()


def test_python_postprocess_and_grocery_parity():
    cleaned = detect_and_clean_hallucination("buy milk milk milk\ncheeeese")
    assert "milk milk milk" not in cleaned
    assert score_text_plausibility("buy milk tomorrow", 0.9) >= 0.7

    corrected = correct_text_with_lexicon(
        "Sheels\nChilden\ntootaposls\nmedical Cheese\nparmesans Spend\n(opte'snal)\nkettte\npotatochip\nSood platanos (gun)",
        capture_mode="grocery",
    )
    assert "Sheets" in corrected
    assert "Children" in corrected
    assert "toothpaste" in corrected
    assert "mexican cheese" in corrected.lower()
    assert "parmesan shred" in corrected.lower()
    assert "(optional)" in corrected
    assert "kettle" in corrected
    assert "potato chip" in corrected
    assert "good platanos (green)" in corrected.lower()
    assert get_parseq_engine().is_available() in (True, False)


def test_remove_ruled_lines_tilted_preserves_crossbars():
    h, w = 140, 420
    canvas = np.full((h, w), 252, dtype=np.uint8)
    # Draw faint broken 2.4-degree tilted notebook rules (2px thick, gray ~185 -> inv ~67 <= 110)
    slope = math.tan(math.radians(2.4))
    for base_y in (52, 96):
        for x in range(10, w - 10):
            if (x % 24) < 3:
                continue  # broken 3px rule gaps
            y = int(round(base_y + slope * (x - w * 0.5)))
            for dy in (0, 1):
                if 0 <= y + dy < h:
                    canvas[y + dy, x] = 185

    # Draw dark text "butter toast" with horizontal crossbars crossing the upper rule
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    font = None
    for fp in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"):
        if Path(fp).is_file():
            font = ImageFont.truetype(fp, 28)
            break
    draw.text((40, 28), "butter toast", fill=25, font=font)
    img_gray = np.array(pil)

    cleaned = remove_ruled_lines(img_gray)
    # Right-side empty rule region (x > 260 around y=96) should be cleanly suppressed
    rule_band = cleaned[85:110, 260:400]
    assert int(np.count_nonzero(rule_band < 210)) == 0
    # Dark ink pixels of "butter toast" (including t crossbars) should remain intact
    text_region = cleaned[25:65, 35:230]
    assert int(np.count_nonzero(text_region < 90)) > 180


def test_strip_spiral_and_right_clipped_preserves_right_words():
    h, w = 400, 440
    binary = np.full((h, w), 255, dtype=np.uint8)
    # Periodic left spiral coils touching x <= 3 on lower half + inset on upper half
    for idx, cy in enumerate(range(35, 370, 36)):
        x_start = 0 if idx >= 4 else 6
        cv2.ellipse(binary, (x_start + 10, cy), (10, 7), 0, 0, 360, 0, 2)

    # Interior main-column text row matching upper & lower bands
    cv2.rectangle(binary, (120, 65), (260, 88), 0, -1)
    cv2.rectangle(binary, (120, 175), (240, 198), 0, -1)
    # Legitimate end-of-line word ("tomatoes") near right side with narrow gap to "Sundried"
    cv2.rectangle(binary, (250, 250), (335, 274), 0, -1)  # "Sundried"
    cv2.rectangle(binary, (355, 250), (425, 274), 0, -1)  # "tomatoes" (gap=20 < chasm)
    # Clipped right-edge second-column fragment touching right border after wide chasm
    cv2.rectangle(binary, (412, 175), (439, 196), 0, -1)

    cleaned = strip_spiral_and_corner_artifacts(binary)
    # Left spiral spine should be completely removed
    assert int(np.count_nonzero(cleaned[:, :28] == 0)) == 0
    # Clipped right-edge fragment at y=175..196 should be removed
    assert int(np.count_nonzero(cleaned[170:202, 405:] == 0)) == 0
    # Legitimate right-side word "tomatoes" should be preserved
    assert int(np.count_nonzero(cleaned[250:275, 355:426] == 0)) > 100


def test_run_preprocessing_parseq_gray_matches_binary_on_dpi_upscale():
    img = Image.new("RGB", (480, 360), (252, 252, 252))
    draw = ImageDraw.Draw(img)
    # Add periodic left spiral coils
    for cy in range(30, 340, 34):
        draw.ellipse((2, cy - 6, 22, cy + 6), outline=(25, 25, 25), width=2)
    # Small text (~14-18px tall) to trigger dpi_scale > 1.0
    font = None
    for fp in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"):
        if Path(fp).is_file():
            font = ImageFont.truetype(fp, 16)
            break
    for idx, text in enumerate(("bread", "milk", "eggs", "spinach")):
        draw.text((80, 40 + idx * 60), text, fill=(20, 20, 20), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    prep = run_preprocessing(buf.getvalue(), ocr_profile="handwriting", export_stages=False)
    assert prep.dpi_scale > 1.0
    assert prep.parseq_gray.shape == prep.binary.shape
    assert prep.contrast_gray.shape == prep.binary.shape
    # Removed left spiral region must be painted white (255) on parseq_gray
    assert int(np.count_nonzero(prep.parseq_gray[:, :24] < 200)) == 0


def test_strip_line_edge_bleeds_removes_top_descender_preserves_core():
    lh, lw = 48, 160
    binary = np.full((lh, lw), 255, dtype=np.uint8)
    # Top-border descender loop from previous line (y=0..7, cw=14, does not touch mid)
    cv2.rectangle(binary, (22, 0), (36, 7), 0, -1)
    # Core word strokes sitting in middle band (y=15..36)
    cv2.rectangle(binary, (20, 15), (60, 36), 0, -1)
    cv2.rectangle(binary, (72, 16), (125, 35), 0, -1)

    mask = strip_line_edge_bleeds(binary, LineBox(x0=0, y0=0, x1=lw - 1, y1=lh - 1))
    assert int(np.count_nonzero(mask[0:8, 22:37] == 0)) == 0
    assert int(np.count_nonzero(mask[15:37, 20:61] == 0)) > 400
    assert int(np.count_nonzero(mask[16:36, 72:126] == 0)) > 400


def test_hallucination_and_plausibility_reject_trocr_prose_and_dates():
    assert score_text_plausibility("after the success", 0.258) < 0.45
    assert score_text_plausibility("buy milk tomorrow", 0.90) >= 0.70

    for bad in (
        "May 20 2021 May 20 2021",
        "Cecrezo's first appearance in the Spanish Spanish",
        "1953 Southern State",
        "2 References",
        "I : - :",
        "- thu",
        "- a m",
        "IK - :",
    ):
        assert detect_and_clean_hallucination(bad) == "", f"Expected rejection for {bad!r}"

    assert detect_and_clean_hallucination("buy milk tomorrow") == "buy milk tomorrow"
    assert detect_and_clean_hallucination("- tea") == "- tea"


def test_parseq_first_handwriting_and_span_merge(monkeypatch: pytest.MonkeyPatch):
    # Gate checks on _accept_trocr_candidate
    narrow_box = LineBox(x0=10, y0=10, x1=90, y1=50)  # aspect = 81/41 = 1.98 < 3.5
    assert _accept_trocr_candidate("after the success", 0.26, narrow_box) == ""
    assert _accept_trocr_candidate("after the success", 0.85, narrow_box) == ""  # >= 3 words on narrow crop
    assert _accept_trocr_candidate("May 20 2021 May 20 2021", 0.92, LineBox(0, 0, 300, 40)) == ""
    assert _accept_trocr_candidate("eggs", 0.88, narrow_box) == "eggs"

    if not get_parseq_engine().is_available():
        pytest.skip("Offline PARSeq model not present")

    # Build a synthetic handwriting image with "milk" and verify PARSeq-first ignores TrOCR hallucination
    img = Image.new("RGB", (460, 140), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = None
    for fp in (r"C:\Windows\Fonts\segoepr.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if Path(fp).is_file():
            font = ImageFont.truetype(fp, 36)
            break
    draw.text((40, 40), "milk", fill=(20, 20, 20), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    # Sabotage TrOCR to emit a low-conf hallucination; PARSeq-first must ignore it
    trocr = get_engine()
    monkeypatch.setattr(trocr, "recognize_line", lambda *_a, **_kw: ("after the success", 0.26))
    res = recognize_image_bytes(buf.getvalue(), ocr_profile="handwriting", export_stages=False)
    assert "after the success" not in res["text"].lower()
    assert "milk" in res["text"].lower()

    # Verify adjacent-span merge disambiguation merges split "as  paragus" -> "asparagus"
    img_split = Image.new("RGB", (460, 140), (255, 255, 255))
    draw_split = ImageDraw.Draw(img_split)
    split_font = (
        ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 36)
        if Path(r"C:\Windows\Fonts\arial.ttf").is_file()
        else font
    )
    draw_split.text((40, 40), "as  paragus", fill=(20, 20, 20), font=split_font)
    buf_split = io.BytesIO()
    img_split.save(buf_split, format="PNG")
    res_split = recognize_image_bytes(buf_split.getvalue(), ocr_profile="handwriting", export_stages=False)
    assert "asparagus" in res_split["text"].lower()


def test_real_notebook_grocery_list_no_hallucinations():
    ref_image = Path(
        r"C:\Users\paste\.gemini\antigravity\brain\a39037d9-3bf5-4972-8be5-9007b443c493\.user_uploaded\media_1795289813276_c93e39f5.jpg"
    )
    if not ref_image.is_file():
        pytest.skip("External reference grocery list image not present")
    if not get_parseq_engine().is_available():
        pytest.skip("Offline PARSeq model not present")

    res = recognize_image_bytes(ref_image.read_bytes(), ocr_profile="auto", export_stages=False)
    text_lower = res["text"].lower()

    for hallucination in (
        "after the success",
        "cecrezo",
        "may 20 2021",
        "1953 southern state",
        "2 references",
        "- thu",
    ):
        assert hallucination not in text_lower, f"Found hallucination {hallucination!r} in {res['text']!r}"

    for expected in (
        "bread",
        "milk",
        "crisps",
        "eggs",
        "lunch for tomorrow",
        "salmon",
        "asparagus",
        "bacon",
        "turkey",
        "spinach",
        "turkey mince",
        "sundried tomatoes",
    ):
        assert expected in text_lower, f"Missing {expected!r} in {res['text']!r}"
    assert ("pain au chocolat" in text_lower) or ("pan au chocolat" in text_lower)
    assert ("broccoli" in text_lower) or ("brocolli" in text_lower)


def test_adaptive_binary_preserves_thin_cursive_strokes_and_crossings():
    """Hysteresis Sauvola + bilateral filter must keep thin 1-2px cursive loops intact across faint rules."""
    img = np.full((120, 260), 245, dtype=np.uint8)
    # Faint horizontal notebook rule
    cv2.line(img, (0, 62), (259, 62), 200, 1)
    # Draw a thin 1-2px cursive-like loop crossing the rule (core dark, top arch fainter)
    cv2.ellipse(img, (130, 60), (24, 18), 0, 0, 360, 145, 1)
    cv2.line(img, (106, 48), (106, 78), 70, 2)
    cv2.line(img, (154, 48), (154, 78), 70, 2)

    cleaned = remove_ruled_lines(img)
    contrast = enhance_contrast(cleaned)
    binary = binarize_adaptive(contrast, window=25, k=0.18)

    # The loop + stems should remain a single connected foreground ink region (no severe fragmentation)
    ink = (binary == 0).astype(np.uint8)
    assert int(np.count_nonzero(ink)) >= 80
    # Top arch of ellipse at y=42 should still have ink pixels
    assert int(np.count_nonzero(ink[38:48, 115:145])) > 0


def test_grocery_autocorrect_prevents_false_swaps_in_auto_mode():
    """Auto-mode grocery correction must never snap 'tomorrow' -> 'onion' or 'Sundried tomatoes' -> 'hundred'."""
    sample = "milk\neggs\nPan a chocolat\nlunch for tormw\nlunch for tomorrow\nhundred parshel\nSundied parshel"
    out = correct_text_with_lexicon(sample, capture_mode="auto")
    lower = out.lower()
    assert "onion" not in lower, f"Unexpected 'onion' swap in {out!r}"
    assert "hundred" not in lower, f"Unexpected 'hundred' swap in {out!r}"
    assert "pain au chocolat" in lower
    assert lower.count("lunch for tomorrow") == 2
    assert lower.count("sundried tomatoes") == 2


def test_grocery_lexicon_json_is_actively_loaded():
    """Verify python_ocr/grocery_lexicon.json is loaded into GROCERY_CORE_WORDS at runtime."""
    from python_ocr.postprocess import GROCERY_CORE_SET

    assert len(GROCERY_CORE_SET) >= 400
    for item in ("zucchini", "quinoa", "sriracha", "croissant", "cilantro", "avocados"):
        assert item in GROCERY_CORE_SET
    assert "zucchini" in correct_text_with_lexicon("milk\neggs\nzuchini", capture_mode="auto").lower()


def test_wikipedia_sidebar_mutation_and_bilingual_grocery_fixes():
    """Mutated Wikipedia sidebar hallucinations must be dropped; bilingual grocery misreads must repair."""
    wiki_leak = "Help learn to editCommunity total recent changesUpload fill"
    assert detect_and_clean_hallucination(wiki_leak) == ""

    sample = (
        "uhip cream\n"
        "Children touch pasta\n"
        "cheddar S9\n"
        "test Uhuevis\n"
        "Kettle potato cup\n"
        "Soosplatanos (gun)"
    )
    out = correct_text_with_lexicon(sample, capture_mode="auto")
    low = out.lower()
    assert "whip cream" in low
    assert "children toothpaste" in low
    assert "cheddar sq." in low
    assert "huevos" in low
    assert "kettle potato chip" in low
    assert "good platanos (green)" in low


def test_parseq_topk_beam_rescores_uncertain_letter_to_lexicon():
    """Top-K logit beam decoding must promote a high-probability 2nd-choice lexicon match over greedy noise."""
    eng = get_parseq_engine()
    if not eng.is_available():
        pytest.skip("Offline PARSeq model not present")
    eng._ensure_loaded()
    assert eng._cfg is not None
    is_itos = "itos" in eng._cfg
    if is_itos:
        itos = list(eng._cfg["itos"])
        v_size = len(itos)
        eos_idx = itos.index("[E]")
        idx_of = lambda ch: itos.index(ch)
    else:
        vocab = eng._cfg["vocab"]
        v_size = len(vocab) + 1
        eos_idx = len(vocab)
        idx_of = lambda ch: vocab.index(ch)

    # Construct a synthetic probability matrix where greedy is "bakon" (not in lexicon)
    # but position 2 has 0.52 for 'k' and 0.44 for 'c' -> "bacon" (in LEXICON_SET)
    probs = np.full((6, v_size), 1e-4, dtype=np.float32)
    for t, ch in enumerate("bacon"):
        probs[t, idx_of(ch)] = 0.96
    probs[2, idx_of("c")] = 0.44
    probs[2, idx_of("k")] = 0.52
    probs[5, eos_idx] = 0.99
    probs /= probs.sum(axis=-1, keepdims=True)

    greedy_only, _ = eng._decode_probs(probs, is_itos, use_lexicon_beam=False)
    beam_text, _ = eng._decode_probs(probs, is_itos, use_lexicon_beam=True)
    assert greedy_only == "bakon"
    assert beam_text == "bacon"
