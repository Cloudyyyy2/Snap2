from __future__ import annotations

import base64
import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from python_ocr.postprocess import (
    correct_grocery_context,
    correct_text_with_lexicon,
    detect_and_clean_hallucination,
    score_text_plausibility,
)
from python_ocr.preprocess import (
    deslant_grayscale,
    run_preprocessing,
    strip_spiral_and_corner_artifacts,
)
from python_ocr.recognizer import get_engine, get_parseq_engine, recognize_image_bytes
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
