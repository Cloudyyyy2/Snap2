from __future__ import annotations

import base64
import io
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class LineBox:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class PreprocessResult:
    original_gray: np.ndarray
    deskewed_gray: np.ndarray
    flattened_gray: np.ndarray
    no_rules_gray: np.ndarray
    contrast_gray: np.ndarray
    binary: np.ndarray
    width: int
    height: int
    skew_angle: float
    dark_inverted: bool
    perspective_warped: bool
    borders_cropped: bool
    dpi_scale: float
    detected_style: str
    applied_steps: List[str]
    line_boxes: List[LineBox]
    total_ink_pixels: int
    export_dir: Optional[str]
    processed_preview_data_url: str


def decode_base64_image(image_input: str) -> bytes:
    """Decode a data URL or raw base64 image string into bytes."""
    if "," in image_input and image_input.strip().startswith("data:"):
        _, b64 = image_input.split(",", 1)
    else:
        b64 = image_input
    return base64.b64decode(b64)


def load_upright_grayscale(image_bytes: bytes, max_width: int = 1800) -> np.ndarray:
    """Load image bytes with EXIF orientation, composite alpha onto white, and resize."""
    pil_img = Image.open(io.BytesIO(image_bytes))
    pil_img = ImageOps.exif_transpose(pil_img)

    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        rgba = pil_img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        pil_img = Image.alpha_composite(bg, rgba).convert("RGB")
    else:
        pil_img = pil_img.convert("RGB")

    rgb = np.array(pil_img)
    h, w = rgb.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        new_h = max(1, int(round(h * scale)))
        rgb = cv2.resize(rgb, (max_width, new_h), interpolation=cv2.INTER_AREA)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def auto_invert_if_dark_mode(gray: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Invert dark-background notes/screenshots so ink is dark on light paper."""
    median_val = float(np.median(gray))
    if median_val < 110.0:
        return 255 - gray, True
    return gray, False


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def correct_document_perspective(gray: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Detect a skewed paper quad (>= 35% area) and perspective-warp upright."""
    h, w = gray.shape
    if w < 180 or h < 180:
        return gray, False

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray, False

    page_area = float(w * h)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        area = cv2.contourArea(c)
        if area < page_area * 0.35 or area > page_area * 0.98:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            src = _order_quad_points(approx.reshape(4, 2).astype(np.float32))
            w_top = float(np.linalg.norm(src[1] - src[0]))
            w_bot = float(np.linalg.norm(src[2] - src[3]))
            h_left = float(np.linalg.norm(src[3] - src[0]))
            h_right = float(np.linalg.norm(src[2] - src[1]))
            dst_w = max(100, int(round(max(w_top, w_bot))))
            dst_h = max(100, int(round(max(h_left, h_right))))
            dst = np.array(
                [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
                dtype=np.float32,
            )
            mat = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(
                gray, mat, (dst_w, dst_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            return warped, True

    return gray, False


def crop_dark_borders(gray: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Trim dark desk borders around a light paper sheet."""
    h, w = gray.shape
    if w < 160 or h < 160:
        return gray, False

    row_med = np.median(gray, axis=1)
    col_med = np.median(gray, axis=0)
    top = 0
    bottom = h - 1
    left = 0
    right = w - 1

    while top < h * 0.22 and row_med[top] < 85:
        top += 1
    while bottom > h * 0.78 and row_med[bottom] < 85:
        bottom -= 1
    while left < w * 0.22 and col_med[left] < 85:
        left += 1
    while right > w * 0.78 and col_med[right] < 85:
        right -= 1

    if top == 0 and bottom == h - 1 and left == 0 and right == w - 1:
        return gray, False
    if (right - left) < w * 0.45 or (bottom - top) < h * 0.45:
        return gray, False

    return gray[top : bottom + 1, left : right + 1].copy(), True


def estimate_and_deskew(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """Estimate small tilt angle via foreground minAreaRect and rotate upright."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, inv_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(inv_bin > 0))
    if len(coords) < 80:
        return gray, 0.0

    pts = coords[:, ::-1].astype(np.float32)
    rect = cv2.minAreaRect(pts)
    angle = rect[-1]
    if angle < -45.0:
        angle = 90.0 + angle
    elif angle > 45.0:
        angle = angle - 90.0

    if abs(angle) < 0.6 or abs(angle) > 15.0:
        return gray, 0.0

    h, w = gray.shape
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(
        gray, mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, round(float(angle), 2)


def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Normalize uneven phone lighting/shadows via large-kernel background division."""
    h, w = gray.shape
    k = max(31, (min(h, w) // 12) | 1)
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    bg = np.maximum(bg, 1)
    norm = (gray.astype(np.float32) / bg.astype(np.float32)) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def remove_ruled_lines(gray: np.ndarray) -> np.ndarray:
    """Detect and suppress long horizontal notebook rules while preserving strokes."""
    _, w = gray.shape
    inv = 255 - gray
    line_len = max(35, w // 4)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    detected = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel)
    _, mask = cv2.threshold(detected, 25, 255, cv2.THRESH_BINARY)
    if np.count_nonzero(mask) == 0:
        return gray
    return cv2.inpaint(gray, mask, 2, cv2.INPAINT_TELEA)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Denoise gently and apply percentile ink-to-paper stretch (matches enhanceInkContrast)."""
    denoised = cv2.medianBlur(gray, 3)
    p_low = float(np.percentile(denoised, 2.0))
    p_high = float(np.percentile(denoised, 78.0))
    if p_high - p_low < 12.0:
        return denoised
    stretched = ((denoised.astype(np.float32) - p_low) / (p_high - p_low)) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def binarize_adaptive(gray: np.ndarray, window: int = 25, k: float = 0.18) -> np.ndarray:
    """Vectorized Sauvola local thresholding producing 0=ink, 255=paper."""
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, ddepth=-1, ksize=(window, window), borderType=cv2.BORDER_REPLICATE)
    sq_mean = cv2.boxFilter(f * f, ddepth=-1, ksize=(window, window), borderType=cv2.BORDER_REPLICATE)
    var = np.maximum(0.0, sq_mean - mean * mean)
    std = np.sqrt(var)
    thresh = mean * (1.0 + k * ((std / 128.0) - 1.0))
    binary = np.where(f <= thresh, 0, 255).astype(np.uint8)
    return binary


def strip_spiral_and_corner_artifacts(binary: np.ndarray) -> np.ndarray:
    """Remove edge-touching spiral coils, inset spiral columns, corner wedges, and speckles."""
    h, w = binary.shape
    if w < 160 or h < 120:
        return binary.copy()

    inv = (binary == 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
    out = binary.copy()

    left_cands: List[int] = []
    right_cands: List[int] = []
    for idx in range(1, num_labels):
        x, y, cw, ch, area = stats[idx]
        # Remove tiny speckles, huge solid blocks, or thin horizontal rule remnants
        if area <= 3 or (cw > w * 0.45 and ch > h * 0.45) or (cw > w * 0.35 and ch <= 4):
            out[labels == idx] = 255
            continue
        # Remove components touching outer left/right border that are narrow/clipped (spiral rings / clipped next-col)
        if (x <= 3 and cw <= w * 0.17 and ch <= h * 0.18) or (
            x + cw >= w - 4 and cw <= w * 0.14 and ch <= h * 0.18
        ):
            out[labels == idx] = 255
            continue
        if x + cw <= w * 0.18 and cw <= w * 0.14 and ch <= h * 0.12 and area >= 6:
            left_cands.append(idx)
        elif x >= w * 0.85 and cw <= w * 0.14 and ch <= h * 0.12 and area >= 6:
            right_cands.append(idx)

    for cands in (left_cands, right_cands):
        if len(cands) >= 3:
            xs = sorted(centroids[i][0] for i in cands)
            med_x = xs[len(xs) // 2]
            aligned = [i for i in cands if abs(centroids[i][0] - med_x) <= max(12, w * 0.055)]
            if len(aligned) >= 3:
                ys = sorted(centroids[i][1] for i in aligned)
                gaps = [ys[k] - ys[k - 1] for k in range(1, len(ys)) if ys[k] - ys[k - 1] >= 6]
                if len(gaps) >= 2:
                    med_gap = sorted(gaps)[len(gaps) // 2]
                    consistent = sum(1 for g in gaps if abs(g - med_gap) <= max(6, med_gap * 0.5))
                    if consistent >= max(2, len(gaps) // 2):
                        for idx in aligned:
                            out[labels == idx] = 255

    return out


def compute_dpi_scale(binary: np.ndarray) -> float:
    """Recommend upscaling when median foreground component height is small (< 26px)."""
    inv = (binary == 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    heights = [
        int(stats[i, cv2.CC_STAT_HEIGHT])
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= 12 and 6 <= stats[i, cv2.CC_STAT_HEIGHT] <= 120
    ]
    if not heights:
        return 1.0
    heights.sort()
    med_h = heights[len(heights) // 2]
    if med_h < 18:
        return 2.0
    if med_h < 26:
        return 1.5
    return 1.0


def deslant_grayscale(crop_gray: np.ndarray) -> np.ndarray:
    """Shear cursive line crop (-24..+24 deg) to maximize vertical projection variance."""
    h, w = crop_gray.shape[:2]
    if h < 16 or w < 24:
        return crop_gray

    _, inv_bin = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = (inv_bin > 0).astype(np.float32)
    if float(ink.sum()) < 20:
        return crop_gray

    best_angle = 0.0
    best_var = -1.0
    for deg in range(-24, 25, 3):
        shear = math.tan(math.radians(deg))
        mat = np.array([[1.0, shear, -shear * h * 0.5], [0.0, 1.0, 0.0]], dtype=np.float32)
        warped = cv2.warpAffine(ink, mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        col_sum = warped.sum(axis=0)
        var = float(np.var(col_sum))
        if var > best_var:
            best_var = var
            best_angle = float(deg)

    if abs(best_angle) < 3.0:
        return crop_gray

    shear = math.tan(math.radians(best_angle))
    extra_w = int(abs(shear) * h)
    new_w = w + extra_w
    offset_x = max(0.0, -shear * h)
    mat = np.array([[1.0, shear, offset_x], [0.0, 1.0, 0.0]], dtype=np.float32)
    return cv2.warpAffine(
        crop_gray, mat, (new_w, h), flags=cv2.INTER_CUBIC, borderValue=255
    )


def split_wide_line_box(binary: np.ndarray, box: LineBox, max_aspect: float = 11.0) -> List[LineBox]:
    """Split ultra-wide line boxes at deep vertical ink valleys without slicing ligatures."""
    bw = box.x1 - box.x0 + 1
    bh = box.y1 - box.y0 + 1
    if bh <= 0 or (bw / float(bh)) <= max_aspect:
        return [box]

    ink = (binary[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1] == 0).astype(np.int32)
    col_sums = ink.sum(axis=0)
    target_w = int(bh * max_aspect * 0.75)
    splits: List[LineBox] = []
    cur_start = 0

    while (bw - cur_start) > bh * max_aspect:
        search_lo = min(bw - 1, cur_start + int(target_w * 0.55))
        search_hi = min(bw - 1, cur_start + target_w)
        if search_hi <= search_lo + 4:
            break
        sub = col_sums[search_lo:search_hi]
        min_idx = int(np.argmin(sub)) + search_lo
        if col_sums[min_idx] <= 2:
            splits.append(LineBox(x0=box.x0 + cur_start, y0=box.y0, x1=box.x0 + min_idx, y1=box.y1))
            cur_start = min_idx + 1
        else:
            break

    if cur_start < bw - 4:
        splits.append(LineBox(x0=box.x0 + cur_start, y0=box.y0, x1=box.x1, y1=box.y1))
    return splits or [box]


def segment_lines(binary: np.ndarray) -> List[LineBox]:
    """Segment text lines via smoothed horizontal projection peaks & valley splitting."""
    h, w = binary.shape
    if h == 0 or w == 0:
        return []

    ink = (binary == 0).astype(np.float32)
    proj = ink.sum(axis=1)

    # Smooth horizontal projection (radius 2) matching lines.ts
    smooth = np.convolve(proj, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    max_proj = float(np.max(smooth))
    if max_proj < 2.0:
        return []

    peak_threshold = max(2.0, max_proj * 0.14)
    min_peak_dist = max(10, int(round(h * 0.025)))

    raw_peaks = [
        y
        for y in range(1, h - 1)
        if smooth[y] >= peak_threshold and smooth[y] >= smooth[y - 1] and smooth[y] >= smooth[y + 1]
    ]

    peaks: List[int] = []
    for p in raw_peaks:
        if not peaks:
            peaks.append(p)
            continue
        prev = peaks[-1]
        min_between = float(np.min(smooth[prev : p + 1]))
        weaker_peak = min(float(smooth[prev]), float(smooth[p]))
        valley_ratio = 0.48 if (p - prev) >= max(18, int(round(min_peak_dist * 1.8))) else 0.35
        if min_between > weaker_peak * valley_ratio:
            if smooth[p] > smooth[prev]:
                peaks[-1] = p
        else:
            peaks.append(p)

    if not peaks:
        return []

    active_threshold = max(0.5, max_proj * 0.015)
    valleys: List[int] = []
    for i in range(len(peaks) - 1):
        p_a, p_b = peaks[i], peaks[i + 1]
        if p_b <= p_a + 1:
            valleys.append(p_a)
            continue
        seg = smooth[p_a + 1 : p_b]
        min_val = float(np.min(seg))
        idxs = np.where(seg <= min_val + 0.5)[0]
        if len(idxs) > 0:
            valley_mid = p_a + 1 + int((idxs[0] + idxs[-1]) // 2)
        else:
            valley_mid = (p_a + p_b) // 2
        valleys.append(max(p_a, min(p_b - 1, valley_mid)))

    boxes: List[LineBox] = []
    for i, pk in enumerate(peaks):
        upper_limit = 0 if i == 0 else valleys[i - 1] + 1
        lower_limit = h - 1 if i == len(peaks) - 1 else valleys[i]

        y0 = pk
        while y0 > upper_limit and smooth[y0 - 1] >= active_threshold:
            y0 -= 1
        y1 = pk
        while y1 < lower_limit and smooth[y1 + 1] >= active_threshold:
            y1 += 1

        prev_dist = pk - peaks[i - 1] if i > 0 else 10**6
        next_dist = peaks[i + 1] - pk if i < len(peaks) - 1 else 10**6
        local_pitch = min(prev_dist, next_dist)
        pad_y = min(max(6, int(round((y1 - y0 + 1) * 0.28))), int(round(local_pitch * 0.32))) if local_pitch < 10**5 else 10
        py0 = max(upper_limit, y0 - pad_y)
        py1 = min(lower_limit, y1 + pad_y)

        if py1 - py0 < 6:
            continue

        col_sums = ink[py0 : py1 + 1, :].sum(axis=0)
        xs = np.where(col_sums > 0)[0]
        if len(xs) == 0:
            continue
        x0 = max(0, int(xs[0]) - 6)
        x1 = min(w - 1, int(xs[-1]) + 6)
        if x1 - x0 < 8:
            continue
        boxes.extend(split_wide_line_box(binary, LineBox(x0=x0, y0=py0, x1=x1, y1=py1)))

    return boxes


def segment_words_in_line(binary: np.ndarray, gray: np.ndarray, box: LineBox) -> List[np.ndarray]:
    """Segment a line crop into word crops for PARSeq word-level OCR."""
    line_bin = binary[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1]
    line_gray = gray[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1]
    if line_bin.size == 0:
        return []
    bh, bw = line_bin.shape
    min_gap = max(6, int(round(bh * 0.26)))
    col_ink = (line_bin == 0).sum(axis=0)

    words: List[np.ndarray] = []
    in_word = False
    w_start = 0
    gap = 0
    for x in range(bw):
        if col_ink[x] > 0:
            if not in_word:
                in_word = True
                w_start = x
            gap = 0
        elif in_word:
            gap += 1
            if gap >= min_gap:
                w_end = x - gap
                if w_end - w_start >= 4:
                    words.append(line_gray[:, max(0, w_start - 3) : min(bw, w_end + 4)])
                in_word = False
                gap = 0
    if in_word:
        w_end = bw - gap
        if w_end - w_start >= 4:
            words.append(line_gray[:, max(0, w_start - 3) : bw])
    return words


def detect_handwriting(binary: np.ndarray, line_boxes: List[LineBox]) -> str:
    """Classify page style as 'handwriting' or 'printed' from line height variance."""
    if len(line_boxes) <= 1:
        return "handwriting"
    heights = [b.y1 - b.y0 + 1 for b in line_boxes]
    mean_h = sum(heights) / len(heights)
    if mean_h <= 0:
        return "handwriting"
    cv = math.sqrt(sum((h - mean_h) ** 2 for h in heights) / len(heights)) / mean_h
    return "handwriting" if cv > 0.18 or mean_h > 34 else "printed"


def encode_png_data_url(gray: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", gray)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def run_preprocessing(
    image_bytes: bytes,
    ocr_profile: str = "auto",
    export_stages: bool = True,
    export_root_dir: Optional[str] = None,
) -> PreprocessResult:
    """Run full OpenCV preprocessing parity pipeline and optionally export named stage PNGs."""
    applied_steps = ["python-opencv-grayscale"]
    raw_gray = load_upright_grayscale(image_bytes)
    polar_gray, dark_inverted = auto_invert_if_dark_mode(raw_gray)
    if dark_inverted:
        applied_steps.append("color-inversion")

    persp_gray, perspective_warped = correct_document_perspective(polar_gray)
    if perspective_warped:
        applied_steps.append("perspective-dewarp")

    cropped_gray, borders_cropped = crop_dark_borders(persp_gray)
    if borders_cropped:
        applied_steps.append("border-crop")

    deskewed, skew_angle = estimate_and_deskew(cropped_gray)
    if abs(skew_angle) > 0:
        applied_steps.append(f"deskew({skew_angle}deg)")

    flattened = flatten_illumination(deskewed)
    applied_steps.append("illumination-flatten")

    no_rules = remove_ruled_lines(flattened)
    applied_steps.append("ruled-lines-removed")

    contrast = enhance_contrast(no_rules)
    applied_steps.append("clahe-contrast")

    binary = strip_spiral_and_corner_artifacts(binarize_adaptive(contrast))
    applied_steps.append("spiral-artifact-filter")

    dpi_scale = compute_dpi_scale(binary)
    if dpi_scale > 1.0:
        nh, nw = int(round(contrast.shape[0] * dpi_scale)), int(round(contrast.shape[1] * dpi_scale))
        contrast = cv2.resize(contrast, (nw, nh), interpolation=cv2.INTER_CUBIC)
        binary = strip_spiral_and_corner_artifacts(binarize_adaptive(contrast))
        applied_steps.append(f"dpi-upscale({dpi_scale}x)")

    h, w = contrast.shape
    total_ink = int(np.count_nonzero(binary == 0))
    line_boxes = [] if total_ink < 30 else segment_lines(binary)
    detected_style = (
        "handwriting"
        if ocr_profile == "handwriting"
        else "printed"
        if ocr_profile == "printed"
        else detect_handwriting(binary, line_boxes)
    )

    export_dir: Optional[str] = None
    if export_stages:
        root = Path(export_root_dir) if export_root_dir else Path(os.getcwd()) / "ocr-exports"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        run_dir = root / f"run-{stamp}-{ocr_profile}"
        run_dir.mkdir(parents=True, exist_ok=True)
        export_dir = str(run_dir)

        cv2.imwrite(str(run_dir / "01_original.png"), raw_gray)
        cv2.imwrite(str(run_dir / "02_deskewed.png"), deskewed)
        cv2.imwrite(str(run_dir / "03b_background_flattened.png"), flattened)
        cv2.imwrite(str(run_dir / "03c_ruled_lines_removed.png"), no_rules)
        cv2.imwrite(str(run_dir / "04_contrast_enhanced.png"), contrast)
        cv2.imwrite(str(run_dir / "05_adaptive_binary.png"), binary)

        for idx, box in enumerate(line_boxes, start=1):
            crop = contrast[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1]
            if crop.size > 0:
                cv2.imwrite(str(run_dir / f"06_line_{idx:02d}.png"), crop)

    preview_url = encode_png_data_url(contrast)
    return PreprocessResult(
        original_gray=raw_gray,
        deskewed_gray=deskewed,
        flattened_gray=flattened,
        no_rules_gray=no_rules,
        contrast_gray=contrast,
        binary=binary,
        width=w,
        height=h,
        skew_angle=skew_angle,
        dark_inverted=dark_inverted,
        perspective_warped=perspective_warped,
        borders_cropped=borders_cropped,
        dpi_scale=dpi_scale,
        detected_style=detected_style,
        applied_steps=applied_steps,
        line_boxes=line_boxes,
        total_ink_pixels=total_ink,
        export_dir=export_dir,
        processed_preview_data_url=preview_url,
    )
