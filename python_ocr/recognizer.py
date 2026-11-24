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
    LEXICON_SET,
    correct_text_with_lexicon,
    correct_word_with_lexicon,
    detect_and_clean_hallucination,
    score_text_plausibility,
)
from .preprocess import (
    LineBox,
    PreprocessResult,
    WordCrop,
    deslant_grayscale,
    run_preprocessing,
    segment_word_crops_in_line,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODELS_ROOT = Path(os.getcwd()) / "public" / "models"
TROCR_DIR = MODELS_ROOT / "trocr-base-handwritten"
PARSEQ_DIR = MODELS_ROOT / "parseq"

DECODER_START_TOKEN_ID = 2
EOS_TOKEN_ID = 2
MAX_NEW_TOKENS = 48
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
        """Height-normalize crop and add horizontal white padding before 384x384 ViT resize."""
        h, w = crop_gray.shape[:2]
        target_h = 48
        scale = target_h / float(max(1, h))
        nw = max(12, min(960, int(round(w * scale))))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(crop_gray, (nw, target_h), interpolation=interp)
        right_pad = max(24, 240 - nw - 16)
        padded = cv2.copyMakeBorder(resized, 8, 8, 16, right_pad, cv2.BORDER_CONSTANT, value=255)
        final_384 = cv2.resize(padded, (384, 384), interpolation=cv2.INTER_CUBIC)
        rgb = np.stack([final_384, final_384, final_384], axis=0).astype(np.float32)
        rgb = (rgb / 255.0 - 0.5) / 0.5
        return np.expand_dims(rgb, axis=0)

    def recognize_line(self, crop_gray: np.ndarray, max_new_tokens: Optional[int] = None) -> Tuple[str, float]:
        self._ensure_loaded()
        assert self._encoder is not None
        assert self._decoder is not None
        assert self._tokenizer is not None

        h, w = crop_gray.shape[:2]
        if max_new_tokens is not None:
            eff_max_tokens = max(1, min(MAX_NEW_TOKENS, int(max_new_tokens)))
        else:
            eff_max_tokens = 20 if (w / float(max(1, h))) < 3.5 else MAX_NEW_TOKENS

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

        for step in range(eff_max_tokens):
            input_ids = np.array([[generated[-1]]], dtype=np.int64)
            use_cache = np.array([step > 0], dtype=bool)
            feed: Dict[str, Any] = {
                "input_ids": input_ids,
                "encoder_hidden_states": encoder_hidden,
                "use_cache_branch": use_cache,
                **past_kv,
            }
            outputs = self._decoder.run(dec_output_names, feed)
            logits = outputs[0][0, -1, :].astype(np.float32, copy=True)

            # Apply repetition penalty (1.1) to previously generated tokens
            for tid in set(generated[1:]):
                logits[tid] = logits[tid] / 1.1 if logits[tid] > 0 else logits[tid] * 1.1

            # Enforce no_repeat_ngram_size = 3
            if len(generated) >= 3:
                tail0, tail1 = generated[-2], generated[-1]
                for idx_prev in range(len(generated) - 2):
                    if generated[idx_prev] == tail0 and generated[idx_prev + 1] == tail1:
                        logits[generated[idx_prev + 2]] = -1e9

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

        if "itos" in self._cfg:
            # baudm/parseq format: itos[0] == '[E]', normalized with mean=0.5, std=0.5
            itos: List[str] = list(self._cfg["itos"])
            target_h, target_w = self._cfg.get("img_size", [32, 128])
            mean = np.array(self._cfg.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32)[:, None, None]
            std = np.array(self._cfg.get("std", [0.5, 0.5, 0.5]), dtype=np.float32)[:, None, None]
            eos_idx = itos.index("[E]") if "[E]" in itos else 0
            # Direct resize matches torchvision T.Resize((32, 128)) used during PARSeq fine-tuning
            canvas = cv2.resize(crop_gray, (int(target_w), int(target_h)), interpolation=cv2.INTER_CUBIC)
            is_itos = True
        else:
            # OnnxTR format: vocab[0..N-1], eos_idx == N
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
            is_itos = False

        rgb = np.stack([canvas, canvas, canvas], axis=0).astype(np.float32) / 255.0
        norm = ((rgb - mean) / std)[None, :, :, :]

        in_name = self._session.get_inputs()[0].name
        logits = self._session.run(None, {in_name: norm})[0][0]
        chars: List[str] = []
        probs: List[float] = []
        for t in range(logits.shape[0]):
            row = logits[t]
            best_idx = int(np.argmax(row))
            if best_idx == eos_idx:
                break
            exp_r = np.exp(row - float(np.max(row)))
            p = float(exp_r[best_idx] / np.sum(exp_r))
            if is_itos:
                if 0 <= best_idx < len(itos):
                    tok = itos[best_idx]
                    if tok not in ("[E]", "[B]", "[P]"):
                        chars.append(tok)
                        probs.append(p)
            elif best_idx < eos_idx:
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


def _accept_trocr_candidate(text: str, conf: float, box: LineBox) -> str:
    """Gate TrOCR candidates by hallucination check, confidence floor, and short-crop prose guard."""
    cleaned = detect_and_clean_hallucination(text)
    if not cleaned:
        return ""
    if conf < 0.60:
        return ""
    aspect = (box.x1 - box.x0 + 1) / float(max(1, box.y1 - box.y0 + 1))
    if aspect < 3.5 and len(cleaned.split()) >= 3:
        return ""
    return cleaned


def _recognize_line_with_parseq(
    parseq: ParseqOcrEngine, prep: PreprocessResult, box: LineBox
) -> Tuple[str, float]:
    """Recognize a line via PARSeq on crisp parseq_gray with speck filter, span merge, and sanity fallback."""
    word_crops = segment_word_crops_in_line(prep.binary, prep.parseq_gray, box, pad=4)
    if not word_crops:
        line_slice = prep.parseq_gray[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1]
        if line_slice.size == 0:
            return "", 0.0
        word_crops = [
            WordCrop(
                crop_gray=cv2.copyMakeBorder(line_slice, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255),
                box=box,
            )
        ]

    surviving: List[Tuple[WordCrop, str, float]] = []
    for wc in word_crops:
        wt, wc_conf = parseq.recognize_word(wc.crop_gray)
        crop_w = wc.box.x1 - wc.box.x0 + 1
        if (
            not wt
            or (not any(c.isalnum() for c in wt) and len(wt) <= 2)
            or (crop_w < 16 and wc_conf < 0.50)
        ):
            continue
        surviving.append((wc, wt, wc_conf))

    if not surviving:
        return "", 0.0

    line_h = max(1, box.y1 - box.y0 + 1)
    max_merge_gap = int(round(line_h * 0.58))
    resolved_words: List[Tuple[str, float]] = []
    wi = 0
    while wi < len(surviving):
        cur_wc, cur_txt, cur_conf = surviving[wi]
        if wi + 1 < len(surviving):
            nxt_wc, nxt_txt, nxt_conf = surviving[wi + 1]
            gap = nxt_wc.box.x0 - cur_wc.box.x1 - 1
            cur_clean = cur_txt.lower() in LEXICON_SET and len(cur_txt) >= 2
            nxt_clean = nxt_txt.lower() in LEXICON_SET and len(nxt_txt) >= 2
            if 0 <= gap <= max_merge_gap and (not cur_clean or not nxt_clean):
                mx0 = max(0, min(cur_wc.box.x0, nxt_wc.box.x0))
                my0 = max(0, min(cur_wc.box.y0, nxt_wc.box.y0))
                mx1 = min(prep.width - 1, max(cur_wc.box.x1, nxt_wc.box.x1))
                my1 = min(prep.height - 1, max(cur_wc.box.y1, nxt_wc.box.y1))
                merged_slice = prep.parseq_gray[my0 : my1 + 1, mx0 : mx1 + 1]
                if merged_slice.size > 0:
                    merged_crop = cv2.copyMakeBorder(
                        merged_slice, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255
                    )
                    merged_txt, merged_conf = parseq.recognize_word(merged_crop)
                    if (
                        merged_txt
                        and " " not in merged_txt
                        and merged_conf >= 0.72
                        and (
                            (merged_txt.lower() in LEXICON_SET and len(merged_txt) >= 3)
                            or correct_word_with_lexicon(merged_txt)[1]
                        )
                    ):
                        resolved_words.append((merged_txt, merged_conf))
                        wi += 2
                        continue
        resolved_words.append((cur_txt, cur_conf))
        wi += 1

    joined = " ".join(t for t, _ in resolved_words).strip()
    mean_conf = round(float(np.mean([c for _, c in resolved_words])), 3) if resolved_words else 0.0

    aspect = (box.x1 - box.x0 + 1) / float(max(1, box.y1 - box.y0 + 1))
    if len(resolved_words) >= 2 and aspect <= 5.5:
        no_lex_before_corr = not any(t.lower() in LEXICON_SET for t, _ in resolved_words)
        if mean_conf < 0.78 or no_lex_before_corr:
            full_slice = prep.parseq_gray[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1]
            if full_slice.size > 0:
                full_crop = cv2.copyMakeBorder(
                    full_slice, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255
                )
                full_txt, full_conf = parseq.recognize_word(full_crop)
                full_clean = detect_and_clean_hallucination(full_txt)
                if (
                    full_clean
                    and full_conf >= 0.78
                    and score_text_plausibility(full_clean, full_conf)
                    > score_text_plausibility(joined, mean_conf)
                ):
                    return full_clean, full_conf

    parseq_line = detect_and_clean_hallucination(joined)
    return parseq_line, mean_conf


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

        best_text = ""
        best_conf = 0.0

        if prep.detected_style == "handwriting" and parseq.is_available():
            if use_deslant:
                deslanted_count += 1
            parseq_line, parseq_conf = _recognize_line_with_parseq(parseq, prep, box)
            best_text, best_conf = parseq_line, parseq_conf

            aspect = (box.x1 - box.x0 + 1) / float(max(1, box.y1 - box.y0 + 1))
            has_wide_oov = False
            if aspect > 4.2 and parseq_line:
                corrected_check = correct_text_with_lexicon(parseq_line, capture_mode=capture_mode)
                for tok in corrected_check.split():
                    core_tok = tok.strip(".,;:!?\"'()-").lower()
                    if len(core_tok) >= 10 and core_tok not in LEXICON_SET:
                        has_wide_oov = True
                        break

            parseq_plaus = score_text_plausibility(parseq_line, parseq_conf) if parseq_line else 0.0
            if trocr.is_available() and (
                not parseq_line
                or parseq_conf < 0.62
                or parseq_plaus < 0.62
                or has_wide_oov
            ):
                crop_for_trocr = deslant_grayscale(crop) if use_deslant else crop
                trocr_raw, trocr_conf = trocr.recognize_line(crop_for_trocr)
                trocr_clean = _accept_trocr_candidate(trocr_raw, trocr_conf, box)
                if trocr_clean and (
                    not parseq_line
                    or score_text_plausibility(trocr_clean, trocr_conf) > parseq_plaus + 0.08
                ):
                    best_text, best_conf = trocr_clean, trocr_conf
        else:
            if use_deslant:
                crop_for_trocr = deslant_grayscale(crop)
                deslanted_count += 1
            else:
                crop_for_trocr = cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)

            if trocr.is_available():
                trocr_raw, trocr_conf = trocr.recognize_line(crop_for_trocr)
                trocr_clean = _accept_trocr_candidate(trocr_raw, trocr_conf, box)
                best_text = trocr_clean
                best_conf = trocr_conf if trocr_clean else 0.0

            if parseq.is_available() and (
                not best_text or score_text_plausibility(best_text, best_conf) < 0.65
            ):
                parseq_line, parseq_conf = _recognize_line_with_parseq(parseq, prep, box)
                if parseq_line and (
                    not best_text
                    or score_text_plausibility(parseq_line, parseq_conf)
                    > score_text_plausibility(best_text, best_conf)
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
        # Render final annotated OCR overlay on top of stage 5 (05_adaptive_binary.png)
        overlay = cv2.cvtColor(prep.binary, cv2.COLOR_GRAY2BGR)
        for i, l in enumerate(lines_out):
            b = l["bbox"]
            cv2.rectangle(overlay, (b["x0"], b["y0"]), (b["x1"], b["y1"]), (235, 99, 37), 2)
            label = f"{i + 1}: {l['text']}"
            tx = max(4, b["x0"])
            ty = max(16, b["y0"] - 4)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(overlay, (tx - 2, ty - th - 4), (tx + tw + 4, ty + 2), (235, 99, 37), -1)
            cv2.putText(
                overlay,
                label,
                (tx, ty - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(Path(prep.export_dir) / "07_ocr_text_overlay.png"), overlay)

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
