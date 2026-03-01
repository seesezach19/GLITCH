"""
Liche - Transition Effects for A↔B Boundary Frames
Premium, intentional transitions for 6/6 cadence and rare/legendary hits.
All OpenCV/NumPy only.
"""

from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np


def _ensure_same_shape(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensure both frames have same shape."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    return a, b


def transition_band_wipe(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    mask: np.ndarray,
    t: float,
    intensity: int,
    seed: int = 0,
) -> np.ndarray:
    """
    B reveals through 2-5 horizontal bands that slide in.
    Jagged edges from low-res noise threshold - signal tearing.
    """
    frame_a, frame_b = _ensure_same_shape(frame_a, frame_b)
    h, w = frame_a.shape[:2]
    np.random.seed(seed)
    n_bands = min(5, max(2, 2 + intensity // 3))
    band_h = h // n_bands
    noise_res = 32
    noise = np.random.rand(noise_res, noise_res).astype(np.float32)
    noise_hi = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
    thresh = 0.3 + 0.4 * (1 - t)  # Jagged: more A when t low
    jagged = (noise_hi > thresh).astype(np.float32)
    reveal = np.zeros((h, w), dtype=np.float32)
    for i in range(n_bands):
        y0, y1 = i * band_h, (i + 1) * band_h
        band_t = max(0, min(1, (t * (n_bands + 1) - i) / 1.5))
        reveal[y0:y1, :] = band_t
    reveal = np.clip(reveal * (0.7 + 0.3 * jagged), 0, 1)
    blend = frame_a.astype(np.float32) * (1 - reveal[:, :, np.newaxis]) + frame_b.astype(np.float32) * reveal[:, :, np.newaxis]
    return np.clip(blend, 0, 255).astype(np.uint8)


def transition_diagonal_rip(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    mask: np.ndarray,
    t: float,
    intensity: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Diagonal band opens (thickness ramps), B inside band with rim glow.
    Perfect for abyssal_tear_rare.
    """
    frame_a, frame_b = _ensure_same_shape(frame_a, frame_b)
    h, w = frame_a.shape[:2]
    np.random.seed(seed)
    angle = 0.4 + 0.2 * np.sin(seed % 100)
    y = np.arange(h, dtype=np.float32)[:, np.newaxis]
    x = np.arange(w, dtype=np.float32)[np.newaxis, :]
    cx, cy = w / 2, h / 2
    dist = (x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
    band_thick = (8 + intensity * 3) * (0.5 + 0.5 * t)
    in_band = np.abs(dist) < band_thick
    rim = np.abs(np.abs(dist) - band_thick) < 4
    rim_glow = np.exp(-np.abs(np.abs(dist) - band_thick) / 3).astype(np.float32)
    reveal = in_band.astype(np.float32) + 0.3 * rim_glow
    reveal = np.clip(reveal * (0.3 + 0.7 * t), 0, 1)
    blend = frame_a.astype(np.float32) * (1 - reveal[:, :, np.newaxis]) + frame_b.astype(np.float32) * reveal[:, :, np.newaxis]
    glow = np.array([60, 120, 180], dtype=np.float32)
    blend = blend + rim[:, :, np.newaxis].astype(np.float32) * glow * (0.15 * intensity * t)
    return np.clip(blend, 0, 255).astype(np.uint8)


def transition_slit_scan_swap(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    mask: np.ndarray,
    t: float,
    intensity: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Vertical slices: left = A, right = B, split line moves across.
    Time sliding feel, compresses well.
    """
    frame_a, frame_b = _ensure_same_shape(frame_a, frame_b)
    h, w = frame_a.shape[:2]
    split_x = int(w * t)
    out = frame_b.copy().astype(np.float32)
    if split_x > 0:
        out[:, :split_x] = frame_a[:, :split_x].astype(np.float32)
    feather = max(2, 8 - intensity // 2)
    if split_x > 0 and split_x < w:
        blend_w = min(feather, split_x, w - split_x)
        for i in range(blend_w):
            alpha = i / blend_w
            l = max(0, split_x - blend_w + i)
            r = min(w, split_x + i)
            if l < r:
                out[:, l:r] = (1 - alpha) * frame_a[:, l:r].astype(np.float32) + alpha * frame_b[:, l:r].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def transition_pixel_scramble_patch(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    mask: np.ndarray,
    t: float,
    intensity: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Localized rectangle scrambles into B with blocky pixelation, then resolves.
    Common-safe, skull readable.
    """
    frame_a, frame_b = _ensure_same_shape(frame_a, frame_b)
    h, w = frame_a.shape[:2]
    np.random.seed(seed)
    out = frame_a.copy()
    patch_w = min(w // 3, 60 + intensity * 8)
    patch_h = min(h // 4, 50 + intensity * 6)
    cx, cy = w // 2, h // 2
    px = cx - patch_w // 2 + int(20 * (np.random.rand() - 0.5))
    py = cy - patch_h // 2 + int(15 * (np.random.rand() - 0.5))
    px = max(0, min(px, w - patch_w))
    py = max(0, min(py, h - patch_h))
    patch_a = frame_a[py : py + patch_h, px : px + patch_w]
    patch_b = frame_b[py : py + patch_h, px : px + patch_w]
    block = max(2, 8 - intensity // 2)
    if t < 0.5:
        scramble = patch_a.copy()
        for by in range(0, patch_h, block):
            for bx in range(0, patch_w, block):
                if np.random.rand() < t * 2:
                    scramble[by : by + block, bx : bx + block] = patch_b[by : by + block, bx : bx + block]
        out[py : py + patch_h, px : px + patch_w] = scramble
    else:
        blend = (1 - t) * patch_a.astype(np.float32) + t * patch_b.astype(np.float32)
        out[py : py + patch_h, px : px + patch_w] = np.clip(blend, 0, 255).astype(np.uint8)
    return out


def transition_edge_first_reveal(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    mask: np.ndarray,
    t: float,
    intensity: int,
    seed: int = 0,
) -> np.ndarray:
    """
    B appears first along skull outline (1px ring), then fills inward.
    Expensive, PFP-friendly. Uses subject mask + dilate/erode for rings.
    """
    frame_a, frame_b = _ensure_same_shape(frame_a, frame_b)
    h, w = frame_a.shape[:2]
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    rings = []
    m = mask_bin.copy()
    for _ in range(8):
        dilated = cv2.dilate(m, kernel)
        ring = cv2.subtract(dilated, m)
        rings.append(ring)
        m = dilated
    reveal = np.zeros((h, w), dtype=np.float32)
    n_rings = len(rings)
    for i, r in enumerate(rings):
        ring_t = max(0, min(1, (t * (n_rings + 2) - i) / 1.5))
        reveal = np.maximum(reveal, r.astype(np.float32) / 255.0 * ring_t)
    fill = cv2.dilate(mask_bin, np.ones((15, 15), np.uint8))
    fill_inv = 255 - fill
    fill_t = max(0, t * 1.2 - 0.2)
    reveal = np.maximum(reveal, (fill_inv.astype(np.float32) / 255.0) * fill_t)
    reveal = np.clip(reveal, 0, 1)
    blend = frame_a.astype(np.float32) * (1 - reveal[:, :, np.newaxis]) + frame_b.astype(np.float32) * reveal[:, :, np.newaxis]
    return np.clip(blend, 0, 255).astype(np.uint8)


def transition_voronoi_shatter_swap(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    mask: np.ndarray,
    t: float,
    intensity: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Voronoi cells: subset flips A→B with small offsets. Crystalline fracture.
    1-2 boundary frames only.
    """
    frame_a, frame_b = _ensure_same_shape(frame_a, frame_b)
    h, w = frame_a.shape[:2]
    np.random.seed(seed)
    n_pts = max(12, 30 - intensity)
    pts = np.random.randint(0, max(1, min(h, w)), (n_pts, 2))
    pts[:, 0] = np.clip(pts[:, 0], 0, h - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, w - 1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist_map = np.full((h, w), 1e9, dtype=np.float32)
    id_map = np.zeros((h, w), dtype=np.int32)
    for i, (py, px) in enumerate(pts):
        d = (yy - py) ** 2 + (xx - px) ** 2
        closer = d < dist_map
        dist_map[closer] = d[closer]
        id_map[closer] = i
    out = frame_a.copy().astype(np.float32)
    flip_chance = t * (0.6 + 0.4 * intensity / 10)
    for i in range(n_pts):
        if np.random.rand() > flip_chance:
            continue
        cell = id_map == i
        ox = int(np.clip(np.random.randn() * 2, -4, 4))
        oy = int(np.clip(np.random.randn() * 2, -4, 4))
        src = np.roll(np.roll(frame_b, ox, axis=1), oy, axis=0)
        out[cell] = src[cell]
    return np.clip(out, 0, 255).astype(np.uint8)


def transition_phase_offset_echo(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    mask: np.ndarray,
    t: float,
    intensity: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Blend A and B with offsets (±2-6px), quick decay. "Two skulls" look.
    Great for soul_shatter_legendary.
    """
    frame_a, frame_b = _ensure_same_shape(frame_a, frame_b)
    h, w = frame_a.shape[:2]
    np.random.seed(seed)
    offset = 2 + intensity // 2
    decay = 1.0 - t * 0.7
    a_shifted = np.roll(np.roll(frame_a, offset, axis=1), -offset // 2, axis=0)
    b_shifted = np.roll(np.roll(frame_b, -offset, axis=1), offset // 2, axis=0)
    blend = (1 - decay) * frame_b.astype(np.float32) + decay * 0.5 * (
        a_shifted.astype(np.float32) + b_shifted.astype(np.float32)
    )
    return np.clip(blend, 0, 255).astype(np.uint8)


def transition_palette_snap_posterize(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    mask: np.ndarray,
    t: float,
    intensity: int,
    seed: int = 0,
) -> np.ndarray:
    """
    One boundary frame: posterize to 4-6 luminance levels, then snap to B.
    Signal resync feel. 1 frame only.
    """
    frame_a, frame_b = _ensure_same_shape(frame_a, frame_b)
    if t < 0.5:
        levels = max(4, 8 - intensity // 2)
        gray = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        posterized = (gray // (256 // levels)) * (256 // levels)
        posterized = np.clip(posterized, 0, 255).astype(np.uint8)
        ratio = (posterized.astype(np.float32) + 1) / (gray.astype(np.float32) + 1)
        out = np.clip(frame_a.astype(np.float32) * ratio[:, :, np.newaxis], 0, 255).astype(np.uint8)
    else:
        out = frame_b
    return out


def transition_chroma_dropout_rebound(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    mask: np.ndarray,
    t: float,
    intensity: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Boundary: near-monochrome (drop chroma). Next frame: oversat cyan/magenta rebound.
    VHS-authentic, rare-feeling.
    """
    frame_a, frame_b = _ensure_same_shape(frame_a, frame_b)
    if t < 0.5:
        yuv = cv2.cvtColor(frame_b, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        drop = 0.15 + 0.1 * (1 - t * 2)
        u_new = np.clip(128 + (u.astype(np.float32) - 128) * drop, 0, 255).astype(np.uint8)
        v_new = np.clip(128 + (v.astype(np.float32) - 128) * drop, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(cv2.merge([y, u_new, v_new]), cv2.COLOR_YUV2BGR)
    else:
        yuv = cv2.cvtColor(frame_b, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        boost = 1.0 + min(1.5, intensity * 0.12 * (t - 0.5) * 2)
        u_new = np.clip(128 + (u.astype(np.float32) - 128) * boost, 0, 255).astype(np.uint8)
        v_new = np.clip(128 + (v.astype(np.float32) - 128) * boost, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(cv2.merge([y, u_new, v_new]), cv2.COLOR_YUV2BGR)
    return out


def transition_scanline_gate(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    mask: np.ndarray,
    t: float,
    intensity: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Rolling scanline: above = A, below = B. Moves quickly across 1-2 frames.
    Cheap, deliberate. Good for commons/uncommons.
    """
    frame_a, frame_b = _ensure_same_shape(frame_a, frame_b)
    h, w = frame_a.shape[:2]
    scan_y = int(h * t)
    out = frame_b.copy().astype(np.float32)
    if scan_y > 0:
        out[:scan_y, :] = frame_a[:scan_y, :].astype(np.float32)
    feather = 4
    if scan_y > 0 and scan_y < h:
        blend_h = min(feather, scan_y, h - scan_y)
        for i in range(blend_h):
            alpha = i / blend_h
            y0 = max(0, scan_y - blend_h + i)
            y1 = min(h, scan_y + i)
            if y0 < y1:
                out[y0:y1, :] = (1 - alpha) * frame_a[y0:y1, :].astype(np.float32) + alpha * frame_b[y0:y1, :].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def transition_micro_jitter_rgb(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    mask: np.ndarray,
    t: float,
    intensity: int,
    seed: int = 0,
) -> np.ndarray:
    """
    1 boundary frame: 1-2px jitter + tiny RGB split. Next frame stable in B.
    Hit without chaos.
    """
    frame_a, frame_b = _ensure_same_shape(frame_a, frame_b)
    h, w = frame_b.shape[:2]
    np.random.seed(seed)
    if t < 0.5:
        jx = np.random.randint(-2, 3)
        jy = np.random.randint(-2, 3)
        shifted = np.roll(np.roll(frame_b, jx, axis=1), jy, axis=0)
        b, g, r = cv2.split(shifted)
        split = max(1, intensity // 3)
        out = cv2.merge([
            np.roll(b, -split, axis=1),
            g,
            np.roll(r, split, axis=1),
        ])
    else:
        out = frame_b
    return np.clip(out, 0, 255).astype(np.uint8)


def transition_noise_threshold_crossfade(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    mask: np.ndarray,
    t: float,
    intensity: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Fixed seeded noise map as reveal mask. noise > threshold shows B.
    Ramp threshold across 2 frames. Coherent, designed look.
    """
    frame_a, frame_b = _ensure_same_shape(frame_a, frame_b)
    h, w = frame_a.shape[:2]
    np.random.seed(seed)
    noise = np.random.rand(h, w).astype(np.float32)
    thresh = t
    reveal = (noise > thresh).astype(np.float32)
    blend = frame_a.astype(np.float32) * (1 - reveal[:, :, np.newaxis]) + frame_b.astype(np.float32) * reveal[:, :, np.newaxis]
    return np.clip(blend, 0, 255).astype(np.uint8)


# Registry: name -> function
TRANSITION_FUNCTIONS = {
    "band_wipe": transition_band_wipe,
    "diagonal_rip": transition_diagonal_rip,
    "slit_scan_swap": transition_slit_scan_swap,
    "pixel_scramble_patch": transition_pixel_scramble_patch,
    "edge_first_reveal": transition_edge_first_reveal,
    "voronoi_shatter_swap": transition_voronoi_shatter_swap,
    "phase_offset_echo": transition_phase_offset_echo,
    "palette_snap_posterize": transition_palette_snap_posterize,
    "chroma_dropout_rebound": transition_chroma_dropout_rebound,
    "scanline_gate": transition_scanline_gate,
    "micro_jitter_rgb": transition_micro_jitter_rgb,
    "noise_threshold_crossfade": transition_noise_threshold_crossfade,
}


def apply_transition(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    mask: np.ndarray,
    t: float,
    transition_name: str,
    intensity: int,
    seed: int = 0,
) -> np.ndarray:
    """Apply named transition. Returns frame_b if unknown or intensity 0."""
    if intensity <= 0 or transition_name not in TRANSITION_FUNCTIONS:
        return frame_b
    fn = TRANSITION_FUNCTIONS[transition_name]
    return fn(frame_a, frame_b, mask, t, intensity, seed)


def pick_transition(weights: dict[str, int], seed: int = 0) -> Optional[str]:
    """Pick transition from dict of name->intensity. Returns None if all zero."""
    active = [(k, v) for k, v in weights.items() if v > 0 and k in TRANSITION_FUNCTIONS]
    if not active:
        return None
    random.seed(seed)
    total = sum(v for _, v in active)
    r = random.uniform(0, total)
    for name, w in active:
        r -= w
        if r <= 0:
            return name
    return active[-1][0]
