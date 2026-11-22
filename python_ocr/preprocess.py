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
class WordCrop:
    crop_gray: np.ndarray
    box: LineBox


@dataclass
class PreprocessResult:
    original_gray: np.ndarray
    deskewed_gray: np.ndarray
    flattened_gray: np.ndarray
    no_rules_gray: np.ndarray
    parseq_gray: np.ndarray
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
    """Detect and suppress tilted/broken horizontal notebook rules while preserving strokes."""
    h, w = gray.shape[:2]
    if h < 16 or w < 40:
        return gray

    inv = 255 - gray
    bridged = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, np.ones((1, 5), dtype=np.uint8))
    klen = max(40, int(round(w * 0.25)))
    horiz_k = np.ones((1, klen), dtype=np.uint8)

    max_det = np.zeros_like(inv)
    for deg in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
        if deg == 0.0:
            det = cv2.morphologyEx(bridged, cv2.MORPH_OPEN, horiz_k)
        else:
            shear = math.tan(math.radians(deg))
            mat = np.array([[1.0, 0.0, 0.0], [shear, 1.0, -shear * w * 0.5]], dtype=np.float32)
            warped = cv2.warpAffine(bridged, mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            opened = cv2.morphologyEx(warped, cv2.MORPH_OPEN, horiz_k)
            mat_inv = np.array([[1.0, 0.0, 0.0], [-shear, 1.0, shear * w * 0.5]], dtype=np.float32)
            det = cv2.warpAffine(opened, mat_inv, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        max_det = np.maximum(max_det, det)

    # Also detect thin isolated horizontal rule fragments (without vertical stem attachments)
    thin_cc_mask = np.zeros(gray.shape[:2], dtype=bool)
    faint_bin = ((inv >= 18) & (inv <= 110)).astype(np.uint8)
    n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(faint_bin, connectivity=8)
    min_cc_w = max(24, int(round(w * 0.08)))
    for idx in range(1, n_cc):
        cw = int(cc_stats[idx, cv2.CC_STAT_WIDTH])
        ch = int(cc_stats[idx, cv2.CC_STAT_HEIGHT])
        if cw >= min_cc_w and ch <= 3 and (cw / max(1, ch)) >= 8.0:
            thin_cc_mask[cc_labels == idx] = True

    det_dil = cv2.dilate((max_det >= 12).astype(np.uint8), np.ones((3, 1), dtype=np.uint8), iterations=1) > 0
    rule_mask = (det_dil | thin_cc_mask) & (inv <= 110)

    # Protect dark vertical stroke crossings (e.g. stems intersecting ruled lines)
    vert_above = np.zeros_like(inv)
    vert_above[2:, :] = inv[:-2, :]
    vert_below = np.zeros_like(inv)
    vert_below[:-2, :] = inv[2:, :]
    cross_protect = (inv > 85) & ((vert_above > 65) | (vert_below > 65))
    rule_mask[cross_protect] = False

    mask_u8 = rule_mask.astype(np.uint8) * 255
    if np.count_nonzero(mask_u8) == 0:
        return gray
    return cv2.inpaint(gray, mask_u8, 2, cv2.INPAINT_TELEA)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Denoise gently and apply percentile ink-to-paper stretch (matches enhanceInkContrast)."""
    denoised = cv2.medianBlur(gray, 3)
    p_low = float(np.percentile(denoised, 2.0))
    p_high = float(np.percentile(denoised, 88.0))
    if p_high - p_low < 12.0:
        return denoised
    stretched = ((denoised.astype(np.float32) - p_low) / (p_high - p_low)) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def binarize_adaptive(gray: np.ndarray, window: int = 25, k: float = 0.18) -> np.ndarray:
    """Vectorized Sauvola local thresholding producing 0=ink, 255=paper."""
    win = max(3, int(window) | 1)
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, ddepth=-1, ksize=(win, win), borderType=cv2.BORDER_REPLICATE)
    sq_mean = cv2.boxFilter(f * f, ddepth=-1, ksize=(win, win), borderType=cv2.BORDER_REPLICATE)
    var = np.maximum(0.0, sq_mean - mean * mean)
    std = np.sqrt(var)
    thresh = mean * (1.0 + k * ((std / 128.0) - 1.0))
    binary = np.where(f <= thresh, 0, 255).astype(np.uint8)
    return binary


def strip_spiral_and_corner_artifacts(binary: np.ndarray) -> np.ndarray:
    """Remove edge-touching spiral coils, inset spiral columns, clipped side columns, corner wedges, and speckles."""
    h, w = binary.shape
    if w < 160 or h < 120:
        return binary.copy()

    inv = (binary == 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
    out = binary.copy()
    rejected = np.zeros(num_labels, dtype=bool)

    def _reject(idx: int) -> None:
        if not rejected[idx]:
            rejected[idx] = True
            out[labels == idx] = 255

    # Step A: tiny specks, huge solid blocks, or thin horizontal rule remnants
    for idx in range(1, num_labels):
        x, y, cw, ch, area = [int(v) for v in stats[idx]]
        if area <= 3 or (cw > w * 0.45 and ch > h * 0.45) or (cw > w * 0.35 and ch <= 4):
            _reject(idx)

    # Step B: left & right spiral spine clustering BEFORE border deletion (includes x <= 3 coils)
    left_cands: List[int] = []
    right_cands: List[int] = []
    for idx in range(1, num_labels):
        if rejected[idx]:
            continue
        x, y, cw, ch, area = [int(v) for v in stats[idx]]
        cx = float(centroids[idx][0])
        if (x + cw <= w * 0.22 or cx <= w * 0.18) and cw <= w * 0.16 and ch <= h * 0.14 and area >= 40:
            left_cands.append(idx)
        elif x >= w * 0.82 and cw <= w * 0.14 and ch <= h * 0.14 and area >= 40:
            right_cands.append(idx)

    for side, cands in (("left", left_cands), ("right", right_cands)):
        if len(cands) < 3:
            continue
        cands_sorted = sorted(cands, key=lambda i: float(centroids[i][1]))
        cys = np.array([float(centroids[i][1]) for i in cands_sorted], dtype=np.float64)
        cxs = np.array([float(centroids[i][0]) for i in cands_sorted], dtype=np.float64)

        use_linear = False
        slope = 0.0
        intercept = float(np.median(cxs))
        if len(cands_sorted) >= 5 and float(np.ptp(cys)) > 1.0:
            fit_slope, fit_intercept = np.polyfit(cys, cxs, 1)
            if abs(float(fit_slope)) <= 0.12:
                use_linear = True
                slope = float(fit_slope)
                intercept = float(fit_intercept)

        tol_x = max(18.0, w * 0.06)
        aligned = [
            i
            for i in cands_sorted
            if abs(float(centroids[i][0]) - ((slope * float(centroids[i][1]) + intercept) if use_linear else intercept))
            <= tol_x
        ]
        if len(aligned) < 3:
            continue

        ys = sorted(float(centroids[i][1]) for i in aligned)
        gaps = [ys[k] - ys[k - 1] for k in range(1, len(ys)) if ys[k] - ys[k - 1] >= 18.0]
        if len(gaps) < 3:
            continue
        med_gap = sorted(gaps)[len(gaps) // 2]
        consistent = sum(1 for g in gaps if abs(g - med_gap) <= max(10.0, med_gap * 0.45))
        if consistent < max(3, int(len(gaps) * 0.5)):
            continue

        for idx in aligned:
            _reject(idx)

        if side == "left":
            max_spiral_right = max(int(stats[i, cv2.CC_STAT_LEFT]) + int(stats[i, cv2.CC_STAT_WIDTH]) for i in aligned)
            for idx in range(1, num_labels):
                if rejected[idx]:
                    continue
                x, _, cw, ch, _ = [int(v) for v in stats[idx]]
                if x + cw <= max_spiral_right + 4 and cw <= w * 0.16 and ch <= h * 0.14:
                    _reject(idx)
        else:
            min_spiral_left = min(int(stats[i, cv2.CC_STAT_LEFT]) for i in aligned)
            for idx in range(1, num_labels):
                if rejected[idx]:
                    continue
                x, _, cw, ch, _ = [int(v) for v in stats[idx]]
                if x >= min_spiral_left - 4 and cw <= w * 0.14 and ch <= h * 0.14:
                    _reject(idx)

    # Step C: residual border-touching stubs and corner wedges
    for idx in range(1, num_labels):
        if rejected[idx]:
            continue
        x, y, cw, ch, _ = [int(v) for v in stats[idx]]
        if (x <= 3 and cw <= w * 0.17 and ch <= h * 0.18) or (
            x + cw >= w - 4 and cw <= w * 0.14 and ch <= h * 0.18
        ):
            _reject(idx)
            continue
        if (y <= 2 or y + ch >= h - 3) and (x <= 2 or x + cw >= w - 3) and cw <= w * 0.2 and ch <= h * 0.2:
            _reject(idx)

    # Step D: clipped right-edge second-column fragments (matching cleanup.ts:418-543)
    chasm = max(32, int(round(w * 0.12)))
    cluster_gap = max(14, int(round(h * 0.05)))
    strip_comps = [
        idx
        for idx in range(1, num_labels)
        if not rejected[idx]
        and int(stats[idx, cv2.CC_STAT_LEFT]) >= w * 0.82
        and int(stats[idx, cv2.CC_STAT_WIDTH]) <= w * 0.16
        and int(stats[idx, cv2.CC_STAT_HEIGHT]) <= h * 0.16
    ]
    if strip_comps:
        visited = [False] * len(strip_comps)
        qualifying_clusters: List[dict] = []
        for i in range(len(strip_comps)):
            if visited[i]:
                continue
            visited[i] = True
            cluster = [strip_comps[i]]
            queue = [i]
            head = 0
            while head < len(queue):
                cur_idx = strip_comps[queue[head]]
                head += 1
                cx0 = int(stats[cur_idx, cv2.CC_STAT_LEFT])
                cy0 = int(stats[cur_idx, cv2.CC_STAT_TOP])
                cx1 = cx0 + int(stats[cur_idx, cv2.CC_STAT_WIDTH]) - 1
                cy1 = cy0 + int(stats[cur_idx, cv2.CC_STAT_HEIGHT]) - 1
                for j in range(len(strip_comps)):
                    if visited[j]:
                        continue
                    o_idx = strip_comps[j]
                    ox0 = int(stats[o_idx, cv2.CC_STAT_LEFT])
                    oy0 = int(stats[o_idx, cv2.CC_STAT_TOP])
                    ox1 = ox0 + int(stats[o_idx, cv2.CC_STAT_WIDTH]) - 1
                    oy1 = oy0 + int(stats[o_idx, cv2.CC_STAT_HEIGHT]) - 1
                    gap_x = max(0, cx0 - ox1, ox0 - cx1)
                    gap_y = max(0, cy0 - oy1, oy0 - cy1)
                    if gap_x <= cluster_gap and gap_y <= cluster_gap:
                        visited[j] = True
                        queue.append(j)
                        cluster.append(o_idx)

            cl_x0 = min(int(stats[c, cv2.CC_STAT_LEFT]) for c in cluster)
            cl_y0 = min(int(stats[c, cv2.CC_STAT_TOP]) for c in cluster)
            cl_x1 = max(int(stats[c, cv2.CC_STAT_LEFT]) + int(stats[c, cv2.CC_STAT_WIDTH]) - 1 for c in cluster)
            cl_y1 = max(int(stats[c, cv2.CC_STAT_TOP]) + int(stats[c, cv2.CC_STAT_HEIGHT]) - 1 for c in cluster)
            touches_border = any(
                int(stats[c, cv2.CC_STAT_LEFT]) + int(stats[c, cv2.CC_STAT_WIDTH]) >= w - 6 for c in cluster
            )
            cl_w = cl_x1 - cl_x0 + 1
            cl_h = cl_y1 - cl_y0 + 1
            if not touches_border or cl_w >= w * 0.14 or cl_h > h * 0.28:
                continue

            cluster_set = set(cluster)
            has_close_neighbor = False
            band_interior_ink = 0
            page_interior_ink = 0
            for o_idx in range(1, num_labels):
                if rejected[o_idx] or o_idx in cluster_set:
                    continue
                ox0 = int(stats[o_idx, cv2.CC_STAT_LEFT])
                oy0 = int(stats[o_idx, cv2.CC_STAT_TOP])
                ox1 = ox0 + int(stats[o_idx, cv2.CC_STAT_WIDTH]) - 1
                oy1 = oy0 + int(stats[o_idx, cv2.CC_STAT_HEIGHT]) - 1
                o_area = int(stats[o_idx, cv2.CC_STAT_AREA])
                if ox0 < w * 0.78 and ox1 > w * 0.22:
                    page_interior_ink += o_area
                if oy1 < cl_y0 - 6 or oy0 > cl_y1 + 6:
                    continue
                if ox0 < cl_x0 and ox1 >= cl_x0 - chasm:
                    has_close_neighbor = True
                    break
                if ox1 < cl_x0 - chasm:
                    band_interior_ink += o_area

            if not has_close_neighbor and (band_interior_ink >= 25 or page_interior_ink >= 60):
                qualifying_clusters.append(
                    {
                        "cluster": cluster,
                        "min_y": cl_y0,
                        "max_y": cl_y1,
                        "has_row_interior": band_interior_ink >= 25,
                    }
                )

        if qualifying_clusters:
            skip_multi_row_col = False
            if len(qualifying_clusters) >= 2:
                min_y_all = min(q["min_y"] for q in qualifying_clusters)
                max_y_all = max(q["max_y"] for q in qualifying_clusters)
                row_aligned_count = sum(1 for q in qualifying_clusters if q["has_row_interior"])
                if row_aligned_count >= 2 and (max_y_all - min_y_all) >= h * 0.22:
                    skip_multi_row_col = True
            if not skip_multi_row_col:
                for q in qualifying_clusters:
                    for c_idx in q["cluster"]:
                        _reject(c_idx)

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
    """Split ultra-wide line boxes only at wide inter-column whitespace chasms."""
    bw = box.x1 - box.x0 + 1
    bh = box.y1 - box.y0 + 1
    if bh <= 0 or (bw / float(bh)) <= max_aspect:
        return [box]

    ink = (binary[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1] == 0).astype(np.int32)
    col_sums = ink.sum(axis=0)
    target_w = int(bh * max_aspect * 0.75)
    min_chasm = max(24, int(round(bh * 0.8)))
    splits: List[LineBox] = []
    cur_start = 0

    while (bw - cur_start) > bh * max_aspect:
        search_lo = min(bw - 1, cur_start + int(target_w * 0.45))
        search_hi = min(bw - 1, cur_start + target_w)
        if search_hi <= search_lo + min_chasm:
            break

        # Scan for a contiguous run of >= min_chasm near-zero columns (col_sums <= 1)
        best_run: Optional[Tuple[int, int]] = None
        run_start: Optional[int] = None
        for x in range(search_lo, search_hi + 1):
            if col_sums[x] <= 1:
                if run_start is None:
                    run_start = x
            else:
                if run_start is not None:
                    if x - run_start >= min_chasm and (best_run is None or (x - run_start) > (best_run[1] - best_run[0] + 1)):
                        best_run = (run_start, x - 1)
                    run_start = None
        if run_start is not None:
            if search_hi + 1 - run_start >= min_chasm and (best_run is None or (search_hi + 1 - run_start) > (best_run[1] - best_run[0] + 1)):
                best_run = (run_start, search_hi)

        if best_run is None:
            break

        mid_idx = (best_run[0] + best_run[1]) // 2
        if mid_idx <= cur_start + 3:
            break
        splits.append(LineBox(x0=box.x0 + cur_start, y0=box.y0, x1=box.x0 + mid_idx, y1=box.y1))
        cur_start = mid_idx + 1

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

        # Low-ink noise peak guard
        band = (binary[py0 : py1 + 1, :] == 0).astype(np.uint8)
        total_band_ink = int(np.count_nonzero(band))
        min_band_ink = 12 if (w < 280 or h < 220) else max(40, int(round((py1 - py0 + 1) * 0.9)))
        if total_band_ink < min_band_ink:
            continue
        n_cc, _, cc_stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
        has_glyph = any(
            int(cc_stats[c_idx, cv2.CC_STAT_AREA]) >= 18 and int(cc_stats[c_idx, cv2.CC_STAT_HEIGHT]) >= 6
            for c_idx in range(1, n_cc)
        )
        if not has_glyph and h < 220:
            small_glyphs = sum(
                1
                for c_idx in range(1, n_cc)
                if int(cc_stats[c_idx, cv2.CC_STAT_AREA]) >= 5 and int(cc_stats[c_idx, cv2.CC_STAT_HEIGHT]) >= 4
            )
            has_glyph = small_glyphs >= 4
        if not has_glyph:
            continue

        col_sums = band.sum(axis=0)
        xs = np.where(col_sums > 0)[0]
        if len(xs) == 0:
            continue

        # Outlier-resistant x0, x1 computation (drop tiny far-away rule specks)
        outlier_gap = max(48, int(round((py1 - py0 + 1) * 2.2)))
        runs: List[Tuple[int, int, int, int]] = []
        r_start = int(xs[0])
        r_prev = int(xs[0])
        for x_val in xs[1:]:
            xv = int(x_val)
            if xv - r_prev > outlier_gap:
                r_ink = int(col_sums[r_start : r_prev + 1].sum())
                r_rows = np.where(band[:, r_start : r_prev + 1].sum(axis=1) > 0)[0]
                r_ch = (int(r_rows[-1]) - int(r_rows[0]) + 1) if len(r_rows) > 0 else 0
                runs.append((r_start, r_prev, r_ink, r_ch))
                r_start = xv
            r_prev = xv
        r_ink = int(col_sums[r_start : r_prev + 1].sum())
        r_rows = np.where(band[:, r_start : r_prev + 1].sum(axis=1) > 0)[0]
        r_ch = (int(r_rows[-1]) - int(r_rows[0]) + 1) if len(r_rows) > 0 else 0
        runs.append((r_start, r_prev, r_ink, r_ch))

        max_run_ink = max(r[2] for r in runs)
        if max_run_ink >= 40 and len(runs) > 1:
            kept_runs = [r for r in runs if r[2] > 25 and r[3] > 3]
            if not kept_runs:
                kept_runs = runs
        else:
            kept_runs = runs

        x0 = max(0, min(r[0] for r in kept_runs) - 6)
        x1 = min(w - 1, max(r[1] for r in kept_runs) + 6)
        if x1 - x0 < 8:
            continue
        boxes.extend(split_wide_line_box(binary, LineBox(x0=x0, y0=py0, x1=x1, y1=py1)))

    return boxes


def strip_line_edge_bleeds(binary: np.ndarray, box: LineBox) -> np.ndarray:
    """Mask out ascender/descender bleeds invading top/bottom borders of a LineBox (0=kept ink, 255=paper)."""
    h, w = binary.shape[:2]
    x0 = max(0, box.x0)
    y0 = max(0, box.y0)
    x1 = min(w - 1, box.x1)
    y1 = min(h - 1, box.y1)
    sub = binary[y0 : y1 + 1, x0 : x1 + 1]
    if sub.size == 0:
        return np.full((1, 1), 255, dtype=np.uint8)

    lh, lw = sub.shape
    mask = np.full((lh, lw), 255, dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((sub == 0).astype(np.uint8), connectivity=8)

    mid_top = int(round(lh * 0.28))
    mid_bot = int(round(lh * 0.78))
    iso_gap = max(10, int(round(lh * 0.24)))

    cc_boxes: List[Tuple[int, int, int, int, int, int, int]] = []
    for idx in range(1, num_labels):
        lx0 = int(stats[idx, cv2.CC_STAT_LEFT])
        ly0 = int(stats[idx, cv2.CC_STAT_TOP])
        cw = int(stats[idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        cc_boxes.append((idx, lx0, ly0, lx0 + cw - 1, ly0 + ch - 1, cw, ch, area))

    for pos, (idx, lx0, ly0, lx1, ly1, cw, ch, area) in enumerate(cc_boxes):
        touches_mid = ly1 >= mid_top and ly0 <= mid_bot
        is_upper_bleed_loop = ly0 <= 2 and ly1 < lh * 0.42 and cw > 10
        is_lower_bleed_loop = ly1 >= lh - 3 and ly0 > lh * 0.62 and cw > 10
        is_top_small_descender = ly0 == 0 and ly1 < lh * 0.48 and area <= 85 and not touches_mid

        is_isolated_border_stem = False
        if (ly0 <= 1 or ly1 >= lh - 2) and cw <= 6 and ch >= lh * 0.45 and (cw / max(1, ch)) < 0.28:
            has_adj = any(
                j != pos and o_lx1 >= lx0 - iso_gap and o_lx0 <= lx1 + iso_gap
                for j, (_, o_lx0, _, o_lx1, _, _, _, _) in enumerate(cc_boxes)
            )
            if not has_adj:
                is_isolated_border_stem = True

        if (
            not is_isolated_border_stem
            and not is_upper_bleed_loop
            and not is_lower_bleed_loop
            and not is_top_small_descender
            and (touches_mid or area <= 28)
        ):
            mask[labels == idx] = 0

    return mask


def segment_word_crops_in_line(
    binary: np.ndarray, gray: np.ndarray, box: LineBox, pad: int = 4
) -> List[WordCrop]:
    """Segment a line crop into white-padded WordCrops with edge bleeds stripped and adaptive word spacing."""
    h, w = binary.shape[:2]
    x0 = max(0, box.x0)
    y0 = max(0, box.y0)
    x1 = min(w - 1, box.x1)
    y1 = min(h - 1, box.y1)
    sub_bin = binary[y0 : y1 + 1, x0 : x1 + 1]
    sub_gray = gray[y0 : y1 + 1, x0 : x1 + 1]
    if sub_bin.size == 0 or sub_gray.size == 0:
        return []

    clamped_box = LineBox(x0=x0, y0=y0, x1=x1, y1=y1)
    mask = strip_line_edge_bleeds(binary, clamped_box)
    lh, lw = mask.shape
    col_ink = (mask == 0).sum(axis=0)
    total_line_ink = int(col_ink.sum())
    if total_line_ink < 14:
        return []

    def _has_faint_gray_in_gap(gx_start: int, gx_end: int) -> bool:
        gap_w = gx_end - gx_start + 1
        if gap_w <= 0 or gap_w > max(18, int(round(lh * 0.45))):
            return False
        band_top = int(round(lh * 0.18))
        band_bot = min(lh - 1, int(round(lh * 0.84)))
        if band_bot < band_top:
            return False
        patch = sub_gray[band_top : band_bot + 1, gx_start : gx_end + 1]
        faint_cols = int(np.count_nonzero(np.any(patch < 226, axis=0)))
        return (faint_cols / float(gap_w)) >= 0.28

    gaps: List[int] = []
    in_gap_ink = False
    gap_start_x = -1
    for i in range(lw):
        if col_ink[i] > 0:
            if not in_gap_ink:
                in_gap_ink = True
                if gap_start_x != -1:
                    gl = i - gap_start_x
                    if gl > 0:
                        gaps.append(gl)
        elif in_gap_ink:
            in_gap_ink = False
            gap_start_x = i

    word_gap = max(9, int(round(lh * 0.27)))
    if len(gaps) >= 4:
        sorted_gaps = sorted(gaps)
        med_gap = sorted_gaps[len(sorted_gaps) // 2]
        p80_gap = sorted_gaps[int(len(sorted_gaps) * 0.8)]
        if p80_gap >= med_gap * 2.2 and p80_gap >= 6:
            word_gap = max(
                7,
                min(int(round(lh * 0.34)), max(int(round(med_gap * 2.2)), int(round(lh * 0.21)))),
            )

    raw_spans: List[Tuple[int, int, int]] = []
    s = -1
    e = -1
    z = 0
    span_ink = 0
    for i in range(lw):
        c_val = int(col_ink[i])
        if c_val > 0:
            if s == -1:
                s = i
                span_ink = c_val
            elif z >= word_gap and not _has_faint_gray_in_gap(e + 1, i - 1):
                raw_spans.append((s, e, span_ink))
                s = i
                span_ink = c_val
            else:
                span_ink += c_val
            e = i
            z = 0
        elif s != -1:
            z += 1
    if s != -1:
        raw_spans.append((s, e, span_ink))

    min_span_ink = max(14, int(round(total_line_ink * 0.04)))
    kept_spans = [sp for sp in raw_spans if sp[2] >= min_span_ink]

    crops: List[WordCrop] = []
    for sx, ex, _ in kept_spans:
        ys, xs = np.where(mask[:, sx : ex + 1] == 0)
        if len(xs) == 0:
            continue
        l_min_x = sx + int(xs.min())
        l_max_x = sx + int(xs.max())
        l_min_y = int(ys.min())
        l_max_y = int(ys.max())

        crop_slice = sub_gray[l_min_y : l_max_y + 1, l_min_x : l_max_x + 1].copy()
        bleed_mask = (
            (sub_bin[l_min_y : l_max_y + 1, l_min_x : l_max_x + 1] == 0)
            & (mask[l_min_y : l_max_y + 1, l_min_x : l_max_x + 1] == 255)
        ).astype(np.uint8) * 255
        if np.count_nonzero(bleed_mask) > 0:
            bleed_dil = cv2.dilate(bleed_mask, np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
            crop_slice[bleed_dil] = 255

        padded = cv2.copyMakeBorder(crop_slice, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        crops.append(
            WordCrop(
                crop_gray=padded,
                box=LineBox(x0=x0 + l_min_x, y0=y0 + l_min_y, x1=x0 + l_max_x, y1=y0 + l_max_y),
            )
        )

    return crops


def segment_words_in_line(binary: np.ndarray, gray: np.ndarray, box: LineBox) -> List[np.ndarray]:
    """Segment a line crop into word crops for PARSeq word-level OCR."""
    return [wc.crop_gray for wc in segment_word_crops_in_line(binary, gray, box)]


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
    parseq_gray = no_rules.copy()
    applied_steps.append("ruled-lines-removed")

    contrast = enhance_contrast(no_rules)
    applied_steps.append("clahe-contrast")

    raw_bin = binarize_adaptive(contrast, window=25, k=0.18)
    binary = strip_spiral_and_corner_artifacts(raw_bin)
    removed_mask = ((raw_bin == 0) & (binary == 255)).astype(np.uint8) * 255
    if np.count_nonzero(removed_mask) > 0:
        removed_dil = cv2.dilate(removed_mask, np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
        contrast[removed_dil] = 255
        parseq_gray[removed_dil] = 255
    applied_steps.append("spiral-artifact-filter")

    dpi_scale = compute_dpi_scale(binary)
    if dpi_scale > 1.0:
        nh, nw = int(round(contrast.shape[0] * dpi_scale)), int(round(contrast.shape[1] * dpi_scale))
        contrast = cv2.resize(contrast, (nw, nh), interpolation=cv2.INTER_CUBIC)
        parseq_gray = cv2.resize(parseq_gray, (nw, nh), interpolation=cv2.INTER_CUBIC)
        win = max(25, int(round(25 * dpi_scale)) | 1)
        raw_bin = binarize_adaptive(contrast, window=win, k=0.18)
        binary = strip_spiral_and_corner_artifacts(raw_bin)
        removed_mask = ((raw_bin == 0) & (binary == 255)).astype(np.uint8) * 255
        if np.count_nonzero(removed_mask) > 0:
            removed_dil = cv2.dilate(removed_mask, np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
            contrast[removed_dil] = 255
            parseq_gray[removed_dil] = 255
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
                padded_crop = cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
                cv2.imwrite(str(run_dir / f"06_line_{idx:02d}.png"), padded_crop)

    preview_url = encode_png_data_url(contrast)
    return PreprocessResult(
        original_gray=raw_gray,
        deskewed_gray=deskewed,
        flattened_gray=flattened,
        no_rules_gray=no_rules,
        parseq_gray=parseq_gray,
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
