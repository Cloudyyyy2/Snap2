from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .postprocess import (
    correct_text_with_lexicon,
    detect_and_clean_hallucination,
    score_text_plausibility,
)
from .preprocess import (
    LineBox,
    deslant_grayscale,
    run_preprocessing,
    segment_words_in_line,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODELS_ROOT = Path(os.getcwd()) / "public" / "models"
TROCR_DIR = MODELS_ROOT / "trocr-base-handwritten"
PARSEQ_DIR = MODELS_ROOT / "parseq"

DECODER_START_TOKEN_ID = 2
EOS_TOKEN_ID = 2
MAX_NEW_TOKENS = 64
NUM_DECODER_LAYERS = 12
NUM_HEADS = 16
HEAD_DIM = 64


class OfflineTrOcrEngine:
    """100% offline ONNX Runtime TrOCR handwriting & printed line recognizer."""

    def __init__(self, model_dir: Path = TROCR_DIR) -> None:
        self.model_dir = model_dir
        self.encoder_path = model_dir / "onnx" / "encoder_model_quantized.onnx"
        self.decoder_path = model_dir / "onnx" / "decoder_model_merged_quantized.onnx"
        self.tokenizer_path = model_dir / "tokenizer.json"
        self._encoder: Optional[ort.InferenceSession] = None
        self._decoder: Optional[ort.InferenceSession] = None
        self._tokenizer: Optional[Tokenizer] = None

    def is_available(self) -> bool:
        return (
            self.encoder_path.is_file()
            and self.decoder_path.is_file()
            and self.tokenizer_path.is_file()
        )

    def _ensure_loaded(self) -> None:
        if not self.is_available():
            raise FileNotFoundError(
                f"Offline TrOCR model files missing under {self.model_dir}. "
                'Run "npm run models:handwriting" while online once.'
            )
        if self._encoder is None:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._encoder = ort.InferenceSession(
                str(self.encoder_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        if self._decoder is None:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._decoder = ort.InferenceSession(
                str(self.decoder_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

    @staticmethod
    def _prepare_pixel_values(crop_gray: np.ndarray) -> np.ndarray:
        """Resize line crop directly to 384x384 RGB with ViT normalization (matches preprocessor_config.json)."""
        resized = cv2.resize(crop_gray, (384, 384), interpolation=cv2.INTER_CUBIC)
        rgb = np.stack([resized, resized, resized], axis=0).astype(np.float32)
        rgb = (rgb / 255.0 - 0.5) / 0.5
        return np.expand_dims(rgb, axis=0)

    def recognize_line(self, crop_gray: np.ndarray) -> Tuple[str, float]:
        self._ensure_loaded()
        assert self._encoder is not None
        assert self._decoder is not None
        assert self._tokenizer is not None

        pixel_values = self._prepare_pixel_values(crop_gray)
        encoder_hidden = self._encoder.run(None, {"pixel_values": pixel_values})[0]

        dec_output_names = [o.name for o in self._decoder.get_outputs()]
        generated: List[int] = [DECODER_START_TOKEN_ID]
        token_probs: List[float] = []

        past_kv: Dict[str, np.ndarray] = {}
        for layer in range(NUM_DECODER_LAYERS):
            past_kv[f"past_key_values.{layer}.decoder.key"] = np.zeros(
                (1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32
            )
            past_kv[f"past_key_values.{layer}.decoder.value"] = np.zeros(
                (1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32
            )
            past_kv[f"past_key_values.{layer}.encoder.key"] = np.zeros(
                (1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32
            )
            past_kv[f"past_key_values.{layer}.encoder.value"] = np.zeros(
                (1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32
            )

        for step in range(MAX_NEW_TOKENS):
            input_ids = np.array([[generated[-1]]], dtype=np.int64)
            use_cache = np.array([step > 0], dtype=bool)
            feed: Dict[str, Any] = {
                "input_ids": input_ids,
                "encoder_hidden_states": encoder_hidden,
                "use_cache_branch": use_cache,
                **past_kv,
            }
            outputs = self._decoder.run(dec_output_names, feed)
            logits = outputs[0][0, -1, :]

            max_logit = float(np.max(logits))
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / np.sum(exp_logits)
            next_id = int(np.argmax(logits))
            if next_id == EOS_TOKEN_ID:
                break

            token_probs.append(float(probs[next_id]))
            generated.append(next_id)

            for idx, name in enumerate(dec_output_names[1:], start=1):
                pkv_name = name.replace("present.", "past_key_values.")
                if step == 0 or ".decoder." in pkv_name:
                    past_kv[pkv_name] = outputs[idx]

        text = self._tokenizer.decode(generated[1:], skip_special_tokens=True).strip()
        conf = float(np.mean(token_probs)) if token_probs else 0.0
        return text, round(conf, 3)


class ParseqOcrEngine:
    """100% offline ONNX Runtime PARSeq word-level recognizer."""

    def __init__(self, model_dir: Path = PARSEQ_DIR) -> None:
        self.model_dir = model_dir
        self.model_path = model_dir / "model.onnx"
        self.config_path = model_dir / "config.json"
        self._session: Optional[ort.InferenceSession] = None
        self._cfg: Optional[Dict[str, Any]] = None

    def is_available(self) -> bool:
        return self.model_path.is_file() and self.config_path.is_file()

    def _ensure_loaded(self) -> None:
        if self._cfg is None:
            self._cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        if self._session is None:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )

    def recognize_word(self, crop_gray: np.ndarray) -> Tuple[str, float]:
        self._ensure_loaded()
        assert self._session is not None
        assert self._cfg is not None

        _, target_h, target_w = self._cfg["input_shape"]  # [3, 32, 128]
        mean = np.array(self._cfg["mean"], dtype=np.float32)[:, None, None]
        std = np.array(self._cfg["std"], dtype=np.float32)[:, None, None]
        vocab: str = self._cfg["vocab"]
        eos_idx = len(vocab)

        h, w = crop_gray.shape[:2]
        scale = min(target_w / max(1, w), target_h / max(1, h))
        nw = max(1, min(target_w, int(round(w * scale))))
        nh = max(1, min(target_h, int(round(h * scale))))
        resized = cv2.resize(crop_gray, (nw, nh), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
        y_off = (target_h - nh) // 2
        x_off = (target_w - nw) // 2
        canvas[y_off : y_off + nh, x_off : x_off + nw] = resized

        rgb = np.stack([canvas, canvas, canvas], axis=0).astype(np.float32) / 255.0
        norm = ((rgb - mean) / std)[None, :, :, :]

        logits = self._session.run(None, {"input": norm})[0][0]  # (33, 127)
        chars: List[str] = []
        probs: List[float] = []
        for t in range(logits.shape[0]):
            row = logits[t]
            best_idx = int(np.argmax(row))
            if best_idx == eos_idx:
                break
            if best_idx < eos_idx:
                exp_r = np.exp(row - float(np.max(row)))
                p = float(exp_r[best_idx] / np.sum(exp_r))
                chars.append(vocab[best_idx])
                probs.append(p)

        text = "".join(chars).strip()
        conf = round(float(np.mean(probs)), 3) if probs else 0.0
        return text, conf


_TROCR_ENGINE = OfflineTrOcrEngine()
_PARSEQ_ENGINE = ParseqOcrEngine()


def get_engine() -> OfflineTrOcrEngine:
    return _TROCR_ENGINE


def get_parseq_engine() -> ParseqOcrEngine:
    return _PARSEQ_ENGINE


def _norm_box(box: LineBox, w: int, h: int) -> Dict[str, float]:
    if w <= 0 or h <= 0:
        return {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
    x = max(0.0, min(1.0, box.x0 / w))
    y = max(0.0, min(1.0, box.y0 / h))
    right = max(x, min(1.0, box.x1 / w))
    bottom = max(y, min(1.0, box.y1 / h))
    return {
        "x": x,
        "y": y,
        "width": max(0.01, right - x),
        "height": max(0.01, bottom - y),
    }


def recognize_image_bytes(
    image_bytes: bytes,
    ocr_profile: str = "auto",
    export_stages: bool = True,
    export_root_dir: Optional[str] = None,
    capture_mode: str = "auto",
) -> Dict[str, Any]:
    """Run full 1:1 parity Python OpenCV preprocessing + ONNX TrOCR/PARSeq + postprocessing."""
    prep = run_preprocessing(
        image_bytes=image_bytes,
        ocr_profile=ocr_profile,
        export_stages=export_stages,
        export_root_dir=export_root_dir,
    )

    if prep.total_ink_pixels < 30 or len(prep.line_boxes) == 0:
        prep.applied_steps.append("no-ink-detected")
        if prep.export_dir:
            (Path(prep.export_dir) / "extracted_text.txt").write_text(
                f"Engine: python-local-ocr\nProfile: {ocr_profile}\nNo readable ink detected.",
                encoding="utf-8",
            )
        return {
            "text": "",
            "lines": [],
            "profile": ocr_profile,
            "detectedStyle": prep.detected_style,
            "recognizer": "python-local-ocr",
            "qualityScore": 0.0,
            "skewAngle": prep.skew_angle,
            "darkInverted": prep.dark_inverted,
            "perspectiveWarped": prep.perspective_warped,
            "bordersCropped": prep.borders_cropped,
            "dpiScale": prep.dpi_scale,
            "appliedSteps": prep.applied_steps,
            "regionsCount": 0,
            "processedPreview": prep.processed_preview_data_url,
            "imageWidth": prep.width,
            "imageHeight": prep.height,
            "needsUserPurpose": True,
            "exportDir": prep.export_dir,
            "exportDirName": Path(prep.export_dir).name if prep.export_dir else None,
        }

    trocr = get_engine()
    parseq = get_parseq_engine()
    use_deslant = prep.detected_style == "handwriting"

    lines_out: List[Dict[str, Any]] = []
    deslanted_count = 0

    for box in prep.line_boxes:
        crop = prep.contrast_gray[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1]
        if crop.size == 0:
            continue
        if use_deslant:
            crop_for_trocr = deslant_grayscale(crop)
            deslanted_count += 1
        else:
            crop_for_trocr = crop

        trocr_text, trocr_conf = trocr.recognize_line(crop_for_trocr)
        trocr_text = detect_and_clean_hallucination(trocr_text)
        best_text, best_conf = trocr_text, trocr_conf

        # Cross-check with PARSeq word-level OCR when available if TrOCR confidence/plausibility is low
        if parseq.is_available() and (
            score_text_plausibility(trocr_text, trocr_conf) < 0.65 or not trocr_text
        ):
            word_crops = segment_words_in_line(prep.binary, prep.contrast_gray, box)
            if word_crops:
                w_texts: List[str] = []
                w_confs: List[float] = []
                for wc in word_crops:
                    wt, wc_conf = parseq.recognize_word(wc)
                    if wt:
                        w_texts.append(wt)
                        w_confs.append(wc_conf)
                parseq_line = detect_and_clean_hallucination(" ".join(w_texts))
                parseq_conf = round(float(np.mean(w_confs)), 3) if w_confs else 0.0
                if parseq_line and score_text_plausibility(parseq_line, parseq_conf) > score_text_plausibility(
                    best_text, best_conf
                ):
                    best_text, best_conf = parseq_line, parseq_conf

        if not best_text:
            continue

        corrected_line = correct_text_with_lexicon(best_text, capture_mode=capture_mode)
        lines_out.append(
            {
                "text": corrected_line,
                "confidence": best_conf,
                "bbox": {"x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1},
                "normBox": _norm_box(box, prep.width, prep.height),
            }
        )

    if deslanted_count > 0:
        prep.applied_steps.append("deslant-lines")
    prep.applied_steps.append("python-lexicon-postprocess")

    full_text = correct_text_with_lexicon(
        "\n".join(l["text"] for l in lines_out).strip(), capture_mode=capture_mode
    )
    avg_conf = round(float(np.mean([l["confidence"] for l in lines_out])), 2) if lines_out else 0.0
    needs_purpose = len(lines_out) == 0 or not full_text

    if prep.export_dir:
        report_lines = [
            "Engine: python-local-ocr (OpenCV + ONNX TrOCR/PARSeq + Python Postprocess)",
            f"Profile: {ocr_profile} | CaptureMode: {capture_mode}",
            f"Detected style: {prep.detected_style}",
            f"Pipeline steps: {' -> '.join(prep.applied_steps)}",
            f"Quality: {avg_conf}",
            "---",
            *[
                f"Line {i + 1} ({int(round(l['confidence'] * 100))}%): {l['text']}"
                for i, l in enumerate(lines_out)
            ],
            "---",
            full_text,
        ]
        (Path(prep.export_dir) / "extracted_text.txt").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )

    return {
        "text": full_text,
        "lines": lines_out,
        "profile": ocr_profile,
        "detectedStyle": prep.detected_style,
        "recognizer": "python-local-ocr",
        "qualityScore": avg_conf,
        "skewAngle": prep.skew_angle,
        "darkInverted": prep.dark_inverted,
        "perspectiveWarped": prep.perspective_warped,
        "bordersCropped": prep.borders_cropped,
        "dpiScale": prep.dpi_scale,
        "appliedSteps": prep.applied_steps,
        "regionsCount": len(prep.line_boxes),
        "processedPreview": prep.processed_preview_data_url,
        "imageWidth": prep.width,
        "imageHeight": prep.height,
        "needsUserPurpose": needs_purpose,
        "exportDir": prep.export_dir,
        "exportDirName": Path(prep.export_dir).name if prep.export_dir else None,
    }
