"""
Liche - Core Glitch Processing Engine
Applies intense glitch effects to masked background regions only.
"""

import cv2
import numpy as np
import random
from typing import Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class GlitchParams:
    """Parameters for all glitch effects."""
    rgb_shift_intensity: int = 0
    chromatic_aberration: int = 0
    scanlines: int = 0
    digital_noise: int = 0
    pixelation: int = 0
    datamosh: int = 0
    melting: int = 0
    vhs_tracking: int = 0
    crt_flicker: int = 0
    bitcrush: int = 0
    tv_static: int = 0
    frame_drift: int = 0
    rolling_scanlines: int = 0
    color_scheme: str = "default"
    mask_sensitivity: float = 0.0
    mask_soft_edge: int = 0
    mask_opacity: float = 1.0  # 1=subject protected, 0=effects full on subject
    foreground_effect: int = 0
    subject_invert: int = 0
    subject_particles: int = 0
    edge_pulse: int = 0
    ghost_trail: int = 0
    block_tear: int = 0
    vignette: int = 0
    bloom: int = 0
    film_grain: int = 0
    pixel_sort: int = 0
    neon_bars: int = 0
    edge_dissolve: int = 0
    chaos_level: float = 0.0
    # New effects
    chroma_dropout: int = 0  # YUV: drop chroma (grey) then oversaturate rebound - VHS feel
    # Spatial/geometry (rare/legendary)
    flow_warp: int = 0       # Displacement field warp - liquid space-time
    slit_scan: int = 0       # Time-smear from past frames at boundary
    parallax_split: int = 0  # Depth layers offset - soul leaving body
    voronoi_shatter: int = 0 # Tessellated Voronoi crystalline fracture
    # Temporal
    echo_crown: int = 0      # Directional echoes with decay, crown/halo legendary
    strobe_phase: int = 0    # Luminance posterize snap at boundary
    frame_drop: int = 0      # Repeat frame + strong hit next - VHS authentic
    # Palette/color
    palette_cycle: int = 0   # Indexed palette rotation
    chroma_collapse: int = 0 # Drop chroma then oversaturate rebound - soul drain
    ordered_dither: int = 0  # Bayer dither overlay - ritual encoded
    # Occult tech
    sigil_ring: int = 0      # Broken ring + tick marks, no runes
    phylactery_glow: int = 0 # Floating orb pulse, chroma bleed
    abyss_window: int = 0    # Portal interior - different space in band
    # Mathy
    fft_jitter: int = 0      # FFT phase jitter - surreal shimmer
    moire_grid: int = 0      # Angled grid drift - cursed signal
    # Transition effects (A↔B boundary frames)
    transition_band_wipe: int = 0
    transition_diagonal_rip: int = 0
    transition_slit_scan_swap: int = 0
    transition_pixel_scramble_patch: int = 0
    transition_edge_first_reveal: int = 0
    transition_voronoi_shatter_swap: int = 0
    transition_phase_offset_echo: int = 0
    transition_palette_snap_posterize: int = 0
    transition_chroma_dropout_rebound: int = 0
    transition_scanline_gate: int = 0
    transition_micro_jitter_rgb: int = 0
    transition_noise_threshold_crossfade: int = 0
    # GLSL shaders
    shader_necrotic_iridescent_flow_intensity: int = 0
    shader_hexagonal_warp_intensity: int = 0
    shader_caustic_flow_intensity: int = 0
    shader_thermal_distort_intensity: int = 0


# Event scheduler: keyframed glitch moments (subtle -> event -> decay)
EVENT_CYCLE = 12
EVENT_PEAK_FRAME = 6


def get_event_scale(frame_idx: int) -> float:
    """Return 0.3-1.0: subtle base, peak at event frame, decay after."""
    pos = frame_idx % EVENT_CYCLE
    if pos < EVENT_PEAK_FRAME:
        return 0.3 + 0.5 * (pos / EVENT_PEAK_FRAME)  # ramp up
    if pos == EVENT_PEAK_FRAME:
        return 1.0
    # decay over remaining frames
    decay_frames = EVENT_CYCLE - EVENT_PEAK_FRAME - 1
    t = (pos - EVENT_PEAK_FRAME - 1) / max(1, decay_frames)
    return 0.5 + 0.5 * (1 - t)  # 1.0 -> 0.5


def vary_params_for_frame(params: GlitchParams, frame_idx: int, total_frames: int = 24) -> GlitchParams:
    """
    Return params with dynamic per-frame variation + event scheduler.
    Event scale: subtle (0.3) -> peak (1.0) at frame 6 -> decay back.
    """
    np.random.seed(frame_idx * 7919)
    random.seed(frame_idx * 7907)
    event = get_event_scale(frame_idx)
    wave = 0.85 + 0.35 * np.sin(frame_idx * 0.7) + 0.15 * np.random.rand()
    wave = np.clip(wave, 0.5, 1.3) * event

    def scale(v: int) -> int:
        if v <= 0:
            return 0
        return max(0, min(10, int(v * wave + np.random.randint(-1, 2))))

    def scale_f(v: float) -> float:
        return float(np.clip(v * (0.8 + 0.4 * np.sin(frame_idx * 0.5)), 0.2, 1.0))

    return GlitchParams(
        rgb_shift_intensity=scale(params.rgb_shift_intensity),
        chromatic_aberration=scale(params.chromatic_aberration),
        scanlines=scale(params.scanlines),
        digital_noise=scale(params.digital_noise),
        pixelation=scale(params.pixelation),
        datamosh=scale(params.datamosh),
        melting=scale(params.melting),
        vhs_tracking=scale(params.vhs_tracking),
        crt_flicker=scale(params.crt_flicker),
        bitcrush=scale(params.bitcrush),
        tv_static=scale(params.tv_static),
        frame_drift=scale(params.frame_drift),
        rolling_scanlines=scale(params.rolling_scanlines),
        color_scheme=params.color_scheme,
        mask_sensitivity=params.mask_sensitivity,
        mask_soft_edge=params.mask_soft_edge,
        mask_opacity=params.mask_opacity,
        foreground_effect=scale(params.foreground_effect),
        subject_invert=scale(params.subject_invert),
        subject_particles=scale(params.subject_particles),
        edge_pulse=scale(params.edge_pulse),
        ghost_trail=scale(params.ghost_trail),
        block_tear=scale(params.block_tear),
        vignette=scale(params.vignette),
        bloom=scale(params.bloom),
        film_grain=scale(params.film_grain),
        pixel_sort=scale(params.pixel_sort),
        neon_bars=scale(params.neon_bars),
        edge_dissolve=scale(params.edge_dissolve),
        chaos_level=scale_f(params.chaos_level),
        chroma_dropout=scale(params.chroma_dropout),
        flow_warp=scale(params.flow_warp),
        slit_scan=scale(params.slit_scan),
        parallax_split=scale(params.parallax_split),
        voronoi_shatter=scale(params.voronoi_shatter),
        echo_crown=scale(params.echo_crown),
        strobe_phase=scale(params.strobe_phase),
        frame_drop=scale(params.frame_drop),
        palette_cycle=scale(params.palette_cycle),
        chroma_collapse=scale(params.chroma_collapse),
        ordered_dither=scale(params.ordered_dither),
        sigil_ring=scale(params.sigil_ring),
        phylactery_glow=scale(params.phylactery_glow),
        abyss_window=scale(params.abyss_window),
        fft_jitter=scale(params.fft_jitter),
        moire_grid=scale(params.moire_grid),
        transition_band_wipe=params.transition_band_wipe,
        transition_diagonal_rip=params.transition_diagonal_rip,
        transition_slit_scan_swap=params.transition_slit_scan_swap,
        transition_pixel_scramble_patch=params.transition_pixel_scramble_patch,
        transition_edge_first_reveal=params.transition_edge_first_reveal,
        transition_voronoi_shatter_swap=params.transition_voronoi_shatter_swap,
        transition_phase_offset_echo=params.transition_phase_offset_echo,
        transition_palette_snap_posterize=params.transition_palette_snap_posterize,
        transition_chroma_dropout_rebound=params.transition_chroma_dropout_rebound,
        transition_scanline_gate=params.transition_scanline_gate,
        transition_micro_jitter_rgb=params.transition_micro_jitter_rgb,
        transition_noise_threshold_crossfade=params.transition_noise_threshold_crossfade,
        shader_necrotic_iridescent_flow_intensity=scale(params.shader_necrotic_iridescent_flow_intensity),
        shader_hexagonal_warp_intensity=scale(params.shader_hexagonal_warp_intensity),
        shader_caustic_flow_intensity=scale(params.shader_caustic_flow_intensity),
        shader_thermal_distort_intensity=scale(params.shader_thermal_distort_intensity),
    )


def apply_rgb_channel_shift(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """RGB channel shifts: sub-pixel stepped (1–3px), band-limited. Rare-tier: shift only highlights (edges)."""
    h, w = frame.shape[:2]
    np.random.seed(frame_idx * 7919)
    # Sub-pixel stepped (1–3px) and band-limited
    base_step = 1 + (intensity % 3)  # 1–3px
    shift_x = int(base_step * (1 + intensity * 0.3) * (0.7 + chaos * 0.6))
    shift_y = int(base_step * 0.5 * (1 + intensity * 0.2) * (0.7 + chaos * 0.6))
    shift_x = max(1, min(shift_x, 12))
    shift_y = max(0, min(shift_y, 6))

    b, g, r = cv2.split(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 150)
    edge_weight = edges.astype(np.float32) / 255.0

    # Rare-tier: intensity >= 5 → shift only on edges (highlights)
    if intensity >= 5:
        edge_weight = np.clip(edge_weight * 1.5, 0, 1)
        edge_3ch = np.stack([edge_weight, edge_weight, edge_weight], axis=-1)
    else:
        edge_3ch = np.ones((h, w, 3), dtype=np.float32)

    M_r = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    M_b = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
    r_shifted = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REPLICATE)
    b_shifted = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REPLICATE)
    glitched = cv2.merge([b_shifted, g, r_shifted])
    blended = (edge_3ch * glitched.astype(np.float32) + (1 - edge_3ch) * frame.astype(np.float32)).astype(np.uint8)
    return np.where(mask[:, :, np.newaxis] == 255, frame, blended).astype(np.uint8)


def apply_chromatic_aberration(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Chromatic aberration: sub-pixel stepped (1–3px), band-limited. Rare-tier: shift only edges."""
    h, w = frame.shape[:2]
    np.random.seed(frame_idx * 12347)
    base_step = 1 + (intensity % 3)
    offset = int(base_step * (1 + intensity * 0.3) * (0.7 + chaos * 0.6))
    offset = max(1, min(offset, 10))

    b, g, r = cv2.split(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 150)
    edge_weight = edges.astype(np.float32) / 255.0

    if intensity >= 5:
        edge_weight = np.clip(edge_weight * 1.5, 0, 1)
        edge_3ch = np.stack([edge_weight, edge_weight, edge_weight], axis=-1)
    else:
        edge_3ch = np.ones((h, w, 3), dtype=np.float32)

    M_pos = np.float32([[1, 0, offset], [0, 1, 0]])
    M_neg = np.float32([[1, 0, -offset], [0, 1, 0]])
    r_shifted = cv2.warpAffine(r, M_pos, (w, h), borderMode=cv2.BORDER_REPLICATE)
    b_shifted = cv2.warpAffine(b, M_neg, (w, h), borderMode=cv2.BORDER_REPLICATE)
    glitched = cv2.merge([b_shifted, g, r_shifted])
    blended = (edge_3ch * glitched.astype(np.float32) + (1 - edge_3ch) * frame.astype(np.float32)).astype(np.uint8)
    return np.where(mask[:, :, np.newaxis] == 255, frame, blended).astype(np.uint8)


def _get_texture_layer(h: int, w: int, frame_idx: int, intensity: int, chaos: float, mode: str) -> np.ndarray:
    """Shared base texture layer. Temporal coherence: noise slides 1px/frame instead of randomizing."""
    np.random.seed(42)
    grain = 2 if mode == "grain" else (3 if mode == "noise" else max(1, 4 - intensity // 4))
    nh, nw = (h + grain) // grain + 2, (w + grain) // grain + 2
    base = np.random.randint(0, 256, (nh, nw), dtype=np.uint8)
    base = cv2.resize(base, (w + 4, h + 4), interpolation=cv2.INTER_NEAREST)
    dy, dx = frame_idx % (h + 4), (frame_idx // (h + 4)) % (w + 4)
    base = np.roll(base, (dy, dx), axis=(0, 1))[:h, :w]
    return cv2.merge([base, base, base]).astype(np.float32)


def _get_coherent_texture(h: int, w: int, token_seed: int, preset_seed: int, frame_idx: int, pan_every_n: int = 2) -> np.ndarray:
    """Deterministic noise texture per token+preset. Pans by 1px every pan_every_n frames instead of regenerating."""
    seed = int((token_seed * 7919 + preset_seed * 31) % (2**32))
    np.random.seed(seed)
    nh, nw = max(8, h // 16), max(8, w // 16)
    base = np.random.randint(0, 256, (nh, nw), dtype=np.uint8)
    base = cv2.resize(base, (w + 4, h + 4), interpolation=cv2.INTER_NEAREST)
    step = frame_idx // pan_every_n
    dy, dx = step % (h + 4), (step // (h + 4)) % (w + 4)
    base = np.roll(base, (dy, dx), axis=(0, 1))[:h, :w]
    return cv2.merge([base, base, base]).astype(np.float32)


def apply_scanlines(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """CRT-style scanlines. Occasional line dropout (missing scanlines)."""
    h, w = frame.shape[:2]
    np.random.seed(frame_idx * 1111)
    line_interval = max(1, 4 - intensity // 3)
    scanline = np.ones((h, w), dtype=np.uint8) * 255
    dark_val = int(255 * (1 - intensity * 0.08))
    for i in range(0, h, line_interval):
        line_y = i
        if line_y >= h:
            break
        if chaos > 0.2 and np.random.rand() < chaos * 0.12:
            continue
        scanline[line_y, :] = dark_val
    scanlines_3ch = cv2.merge([scanline, scanline, scanline])
    glitched = cv2.multiply(frame, scanlines_3ch // 255)
    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def apply_digital_noise(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Digital noise - shared texture layer, temporal coherence (slides 1px/frame)."""
    h, w = frame.shape[:2]
    tex = _get_texture_layer(h, w, frame_idx, intensity, chaos, "noise")
    noise_amount = intensity * 25 + int(chaos * 50)
    noise = ((tex - 128) / 128 * (noise_amount / 2)).astype(np.int32)
    glitched = np.clip(frame.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def apply_pixelation(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Localized pixelation band (not whole-frame) - avoids muddying pixel art."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    block_size = max(2, intensity * 2 + int(chaos * 4))
    np.random.seed(frame_idx * 1111)
    # Localized band: ~25% of height
    band_h = max(block_size * 2, h // 4)
    y0 = np.random.randint(0, max(1, h - band_h))
    y1 = min(y0 + band_h, h)
    glitched = frame.copy()
    band = frame[y0:y1, :].copy()
    small = cv2.resize(band, (w // block_size, (y1 - y0) // block_size), interpolation=cv2.INTER_NEAREST)
    band_pix = cv2.resize(small, (w, y1 - y0), interpolation=cv2.INTER_NEAREST)
    # In band: pixelate only where mask allows (background)
    m = mask[y0:y1, :]
    glitched[y0:y1, :] = np.where(m[:, :, np.newaxis] == 255, frame[y0:y1, :], band_pix)
    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def apply_datamosh(frame: np.ndarray, prev_frame: Optional[np.ndarray], mask: np.ndarray,
                   intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Optical-flow datamosh: rare+ only, mask-protected. Band-limited + short-lived (1–2 frames)."""
    h, w = frame.shape[:2]
    if prev_frame is None:
        return frame

    # Short-lived: only active near event peak (frames 5–7 of 12-cycle)
    pos = frame_idx % EVENT_CYCLE
    if pos < 4 or pos > 8:
        return frame

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.1, flags=0,
    )

    # Band-limited scale so silhouette stays readable
    scale = 0.3 + intensity * 0.08 + chaos * 0.15
    scale = min(scale, 1.2)
    np.random.seed(frame_idx * 7919)
    noise = np.random.randn(h, w, 2).astype(np.float32) * (intensity * 0.3)

    y_coords = np.arange(h, dtype=np.float32)[:, np.newaxis]
    x_coords = np.arange(w, dtype=np.float32)[np.newaxis, :]
    map_x = (x_coords - flow[:, :, 0] * scale + noise[:, :, 0]).astype(np.float32)
    map_y = (y_coords - flow[:, :, 1] * scale + noise[:, :, 1]).astype(np.float32)

    glitched = cv2.remap(prev_frame, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def apply_melting(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Liquid corruption: low-res displacement map so it looks like liquid, not blur. Rare+ only."""
    h, w = frame.shape[:2]
    np.random.seed(frame_idx * 54321)
    # Low-res displacement map (e.g. 16x16) then upsample for coherent "liquid" look
    # Very gentle: intensity 1 = ~0.3px, intensity 5 = ~1.2px, intensity 10 = ~2.5px
    d_res = 16
    disp_h, disp_w = max(4, h // d_res), max(4, w // d_res)
    displacement = intensity * 0.2 + chaos * 0.5
    phase = frame_idx * 0.1
    yy = np.linspace(0, h, disp_h)
    xx = np.linspace(0, w, disp_w)
    low_wave_x = np.sin(np.outer(yy, np.ones(disp_w)) * 0.02 + phase) * displacement
    low_wave_y = np.cos(np.outer(np.ones(disp_h), xx) * 0.02 + phase) * displacement * 0.5
    noise_scale = 0.15 + 0.08 * intensity
    low_noise = np.random.randn(disp_h, disp_w, 2).astype(np.float32) * noise_scale
    low_map_x = np.tile(np.arange(disp_w, dtype=np.float32), (disp_h, 1)) * (w / disp_w) + low_wave_x + low_noise[:, :, 0] * (w / disp_w)
    low_map_y = np.tile(np.arange(disp_h, dtype=np.float32)[:, np.newaxis], (1, disp_w)) * (h / disp_h) + low_wave_y + low_noise[:, :, 1] * (h / disp_h)
    map_x = cv2.resize(low_map_x, (w, h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(low_map_y, (w, h), interpolation=cv2.INTER_LINEAR)
    y_coords = np.arange(h, dtype=np.float32)[:, np.newaxis]
    x_coords = np.arange(w, dtype=np.float32)[np.newaxis, :]
    map_x = np.where(mask < 128, map_x, x_coords).astype(np.float32)
    map_y = np.where(mask < 128, map_y, y_coords).astype(np.float32)
    map_x = np.ascontiguousarray(map_x)
    map_y = np.ascontiguousarray(map_y)
    glitched = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def apply_vhs_tracking(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """VHS tracking errors - horizontal banding. Amplitude 'breathes' (ramps up/down) vs constant jitter."""
    h, w = frame.shape[:2]
    np.random.seed(frame_idx * 8888)
    glitched = frame.copy()

    # Breathe: amplitude ramps up/down over ~8 frames
    breathe = 0.5 + 0.5 * np.sin(frame_idx * 0.4)
    amp = (0.5 + breathe) * intensity * 2

    band_height = max(2, 20 - intensity * 2)
    for y in range(0, h, band_height):
        if mask[y:y+band_height, :].mean() < 200:
            offset = int(np.clip(np.random.randn() * amp, -intensity * 3, intensity * 3))
            if offset != 0:
                glitched[y:y+band_height, :] = np.roll(frame[y:y+band_height, :], offset, axis=1)

    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def apply_crt_flicker(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """CRT flicker - brightness variation. Occasional line dropout (missing scanline-like dark bands)."""
    h, w = frame.shape[:2]
    np.random.seed(frame_idx * 339)
    flicker = 1.0 - (intensity * 0.05) * (0.5 + 0.5 * np.sin(frame_idx * 2) * chaos)
    glitched = (frame.astype(np.float32) * flicker).astype(np.uint8)
    # Occasional dropout: darken a random row band (chaos-driven)
    if chaos > 0.25 and np.random.rand() < chaos * 0.1:
        drop_y = np.random.randint(0, max(1, h - 2))
        glitched[drop_y:drop_y+2, :] = (glitched[drop_y:drop_y+2, :].astype(np.float32) * 0.6).astype(np.uint8)
    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def apply_bitcrush(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Bit depth reduction. Subtle on commons. Rare: bitcrush only background (mask already does this)."""
    # Gentler scaling for common: levels drop less aggressively at low intensity
    levels = max(2, 256 - intensity * 18)
    if intensity <= 0:
        return frame
    glitched = (frame // (256 // levels)) * (256 // levels)
    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def apply_tv_static(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """TV static - shared texture layer, temporal coherence (slides 1px/frame). Intensity modulates blend."""
    h, w = frame.shape[:2]
    static = _get_texture_layer(h, w, frame_idx, intensity, chaos, "static").astype(np.uint8)
    amount = min(0.6, intensity * 0.06)
    glitched = cv2.addWeighted(frame, 1 - amount, static, amount, 0)
    if intensity >= 3:
        jitter = (frame_idx * 7 + intensity * 3) % 5 - 2
        jitter = int(jitter * (intensity / 10))
        if abs(jitter) > 0:
            M = np.float32([[1, 0, jitter], [0, 1, 0]])
            glitched = cv2.warpAffine(glitched, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def apply_film_grain(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Subtle film grain - shared texture layer, temporal coherence. Intensity modulates amplitude."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    tex = _get_texture_layer(h, w, frame_idx, intensity, chaos, "grain")
    amp = intensity * 1.2
    noise = ((tex - 128) / 128 * amp).astype(np.int16)
    glitched = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def apply_pixel_sort(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Pixel sort in small regions or along edges. Rare: sort follows rolling scanline."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    np.random.seed(frame_idx * 54321)
    glitched = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 120)
    # Small regions: process fewer rows, smaller segments
    row_skip = max(2, 5 - intensity // 2)
    scan_y = (frame_idx * 3) % h  # Rolling scanline for rare-tier
    n_rows = (h + row_skip - 1) // row_skip
    for ri in range(n_rows):
        y = ri * row_skip
        if y >= h:
            break
        if mask[y:min(y+row_skip, h), :].mean() >= 220:
            continue
        # Rare: only sort near the rolling scanline
        if intensity >= 5 and abs(y - scan_y) > row_skip * 3:
            continue
        for dy in range(row_skip):
            row_y = y + dy
            if row_y >= h:
                break
            row_bgr = glitched[row_y, :, :].copy()
            row_gray = gray[row_y, :]
            n_segments = max(2, min(intensity + 2, 6))
            seg_len = w // n_segments
            for s in range(n_segments):
                lo, hi = s * seg_len, min((s + 1) * seg_len, w)
                if hi <= lo + 2:
                    continue
                seg = row_bgr[lo:hi]
                seg_gray = row_gray[lo:hi]
                order = np.argsort(seg_gray)
                if np.random.rand() > 0.5:
                    order = order[::-1]
                row_bgr[lo:hi] = seg[order]
            glitched[row_y, :, :] = row_bgr
    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def _refine_color(bgr: np.ndarray) -> np.ndarray:
    """Subtle tint from image color — refined, not garish."""
    b, g, r = float(bgr[0]), float(bgr[1]), float(bgr[2])
    mx = max(b, g, r)
    if mx < 5:
        return np.array([180, 200, 220], dtype=np.float32)  # soft cool gray
    # Gentle push: preserve hue, slight saturation bump, soft luminance
    out = np.array([b, g, r], dtype=np.float32) * (220 / mx)
    out = np.clip(out, 0, 255)
    return out


def _add_sigil_ticks(glitched: np.ndarray, bar_mask: np.ndarray, color: np.ndarray, tick_spacing: int = 12) -> None:
    """Add tiny rune ticks along the bar for LICHE branding. Modifies glitched in-place."""
    h, w = glitched.shape[:2]
    ys, xs = np.where(bar_mask > 0.3)
    if len(ys) < 4:
        return
    yc, xc = int(ys.mean()), int(xs.mean())
    # Horizontal or vertical bar
    if np.ptp(xs) > np.ptp(ys):
        for x in range(0, w, tick_spacing):
            if 1 <= x < w - 2 and 1 <= yc < h - 2:
                glitched[yc:yc+2, x:x+2, :] = (
                    glitched[yc:yc+2, x:x+2, :].astype(np.float32) * 0.6
                    + np.array([color], dtype=np.float32) * 0.4
                )
    else:
        for y in range(0, h, tick_spacing):
            if 1 <= y < h - 2 and 1 <= xc < w - 2:
                glitched[y:y+2, xc:xc+2, :] = (
                    glitched[y:y+2, xc:xc+2, :].astype(np.float32) * 0.6
                    + np.array([color], dtype=np.float32) * 0.4
                )


def apply_neon_bars(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Sigil bars — light-leak with tiny rune ticks for LICHE branding."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    np.random.seed(frame_idx * 8888)
    glitched = frame.copy().astype(np.float32)
    glitch_region = (mask < 128).astype(np.float32)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 120)
    grad_y = np.abs(np.gradient(gray.astype(np.float32), axis=0))
    grad_x = np.abs(np.gradient(gray.astype(np.float32), axis=1))
    weight_h = (np.sum(edges * glitch_region, axis=1) + np.sum(grad_y * glitch_region, axis=1) * 0.5).astype(np.float32)
    weight_w = (np.sum(edges * glitch_region, axis=0) + np.sum(grad_x * glitch_region, axis=0) * 0.5).astype(np.float32)
    weight_h = np.maximum(weight_h, 1) / np.maximum(weight_h.sum(), 1)
    weight_w = np.maximum(weight_w, 1) / np.maximum(weight_w.sum(), 1)

    # Subtle blend — light leak, not solid overlay (0.08–0.25)
    blend_base = 0.06 + intensity * 0.04 + chaos * 0.06
    blend_base = np.clip(blend_base, 0.06, 0.28)
    n_bars = min(intensity + 2, 5)  # Fewer, more intentional
    phase = (frame_idx * 5) % 15

    for i in range(n_bars):
        bar_type = np.random.randint(0, 3)  # horiz, vert, band only
        bw = max(1, np.random.randint(1, 3 + intensity))  # Thinner
        sigma = max(3, bw * 4)  # Soft feather

        if bar_type == 0 and h > bw:
            wh = weight_h[:h - bw]
            if wh.sum() > 0:
                wh = wh / wh.sum()
                by = np.random.choice(h - bw, p=wh)
                by = (by + phase) % max(1, h - bw)
                if mask[by:by + bw, :].mean() < 200:
                    row_pixels = frame[by:by + bw, :][glitch_region[by:by + bw, :] > 0.5]
                    color = np.median(row_pixels, axis=0) if len(row_pixels) > 5 else frame[by + bw // 2, w // 2]
                    color = _refine_color(color)
                    bar_mask = np.zeros((h, w), dtype=np.float32)
                    bar_mask[by:by + bw, :] = 1
                    bar_mask = cv2.GaussianBlur(bar_mask, (0, 0), sigma)
                    bar_mask = bar_mask * glitch_region
                    bar_mask_3 = bar_mask[:, :, np.newaxis]
                    glitched = glitched * (1 - bar_mask_3 * blend_base) + color * (bar_mask_3 * blend_base)
                    _add_sigil_ticks(glitched, bar_mask, color)

        elif bar_type == 1 and w > bw:
            ww = weight_w[:w - bw]
            if ww.sum() > 0:
                ww = ww / ww.sum()
                bx = np.random.choice(w - bw, p=ww)
                bx = (bx + phase) % max(1, w - bw)
                if mask[:, bx:bx + bw].mean() < 200:
                    col_pixels = frame[:, bx:bx + bw][glitch_region[:, bx:bx + bw] > 0.5]
                    color = np.median(col_pixels, axis=0) if len(col_pixels) > 5 else frame[h // 2, bx + bw // 2]
                    color = _refine_color(color)
                    bar_mask = np.zeros((h, w), dtype=np.float32)
                    bar_mask[:, bx:bx + bw] = 1
                    bar_mask = cv2.GaussianBlur(bar_mask, (0, 0), sigma)
                    bar_mask = bar_mask * glitch_region
                    bar_mask_3 = bar_mask[:, :, np.newaxis]
                    glitched = glitched * (1 - bar_mask_3 * blend_base) + color * (bar_mask_3 * blend_base)
                    _add_sigil_ticks(glitched, bar_mask, color)

        else:
            if h > bw * 4:
                n_rows = h - bw * 4
                wh = weight_h[:n_rows]
                if wh.sum() > 0:
                    wh = wh / wh.sum()
                    by = np.random.choice(n_rows, p=wh)
                    bh = min(bw * 4, h - by)
                    if mask[by:by + bh, :].mean() < 200:
                        band_pixels = frame[by:by + bh, :][glitch_region[by:by + bh, :] > 0.5]
                        color = np.median(band_pixels, axis=0) if len(band_pixels) > 10 else frame[by + bh // 2, w // 2]
                        color = _refine_color(color)
                        bar_mask = np.zeros((h, w), dtype=np.float32)
                        bar_mask[by:by + bh, :] = 1
                        ramp = np.linspace(0.2, 1.0, bh)
                        ramp = np.minimum(ramp, ramp[::-1])
                        bar_mask[by:by + bh, :] *= ramp[:, np.newaxis]
                        bar_mask = cv2.GaussianBlur(bar_mask, (0, 0), sigma)
                        bar_mask = bar_mask * glitch_region
                        bar_mask_3 = bar_mask[:, :, np.newaxis]
                        glitched = glitched * (1 - bar_mask_3 * blend_base) + color * (bar_mask_3 * blend_base)
                        _add_sigil_ticks(glitched, bar_mask, color)

    glitched = np.clip(glitched, 0, 255).astype(np.uint8)
    # Very subtle overall softness
    if intensity >= 2:
        blurred = cv2.GaussianBlur(glitched, (3, 3), 0.8).astype(np.float32)
        glitched = np.clip(glitched.astype(np.float32) * 0.97 + blurred * 0.03, 0, 255).astype(np.uint8)
    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def apply_edge_dissolve(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Dissolve background inward (not skull outward). Only corrupt background bands - keeps PFP readable."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    np.random.seed(frame_idx * 9999)
    glitched = frame.copy()
    slice_h = max(2, 12 - intensity)
    # Only offset slices that are mostly background (mask < 128) - dissolve inward, not subject
    for y in range(0, h - slice_h, slice_h):
        if mask[y:y+slice_h, :].mean() > 80:
            continue
        offset = np.random.randint(-intensity * 6, intensity * 6 + 1)
        if offset != 0:
            glitched[y:y+slice_h, :] = np.roll(frame[y:y+slice_h, :].copy(), offset, axis=1)
    return glitched


def apply_frame_drift(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Frame drift - whole image wobbles. Clamp to 1–2px for common/uncommon (intensity 1–4)."""
    h, w = frame.shape[:2]
    raw_x = np.sin(frame_idx * 0.4) * intensity * 4 + np.cos(frame_idx * 0.3) * intensity * 2
    raw_y = np.sin(frame_idx * 0.5 + 1) * intensity * 2 + np.cos(frame_idx * 0.2) * intensity * 3
    clamp = 2 if intensity <= 4 else (4 if intensity <= 7 else 8)
    drift_x = int(np.clip(raw_x, -clamp, clamp))
    drift_y = int(np.clip(raw_y, -clamp, clamp))
    if drift_x != 0 or drift_y != 0:
        M = np.float32([[1, 0, drift_x], [0, 1, drift_y]])
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return frame


def apply_rolling_scanlines(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Scanlines that roll down each frame. Occasional line dropout (missing scanlines)."""
    h, w = frame.shape[:2]
    np.random.seed(frame_idx * 5432)
    line_interval = max(1, 5 - intensity // 2)
    phase = (frame_idx * 3) % line_interval
    scanline = np.ones((h, w), dtype=np.uint8) * 255
    dark_val = int(255 * (1 - intensity * 0.1))
    for i in range(0, h, line_interval):
        line_y = (i + phase) % h
        if chaos > 0.2 and np.random.rand() < chaos * 0.1:
            continue
        scanline[line_y, :] = dark_val
    scanlines_3ch = cv2.merge([scanline, scanline, scanline])
    glitched = cv2.multiply(frame, scanlines_3ch // 255)
    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def apply_color_scheme(frame: np.ndarray, scheme: str, mask: np.ndarray) -> np.ndarray:
    """Apply color grading / tint. Applied to glitch regions (inverse of mask for blending)."""
    h, w = frame.shape[:2]
    result = frame.copy().astype(np.float32)
    b, g, r = cv2.split(result)
    if scheme == "cold":
        b = np.clip(b * 1.2 + 15, 0, 255)
        r = np.clip(r * 0.85, 0, 255)
    elif scheme == "warm":
        r = np.clip(r * 1.2 + 20, 0, 255)
        b = np.clip(b * 0.8, 0, 255)
    elif scheme == "neon":
        b = np.clip(b * 1.3 + (255 - g) * 0.2, 0, 255)
        r = np.clip(r * 1.2 + (255 - b) * 0.2, 0, 255)
        g = np.clip(g * 0.9, 0, 255)
    elif scheme == "inverted":
        result = 255 - result
        return np.where(mask[:, :, np.newaxis] == 255, frame, result.astype(np.uint8))
    elif scheme == "vhs":
        r = np.clip(r * 1.15 + 10, 0, 255)
        g = np.clip(g * 0.95, 0, 255)
        b = np.clip(b * 0.9 - 5, 0, 255)
    elif scheme == "sepia":
        tr = 0.393 * r + 0.769 * g + 0.189 * b
        tg = 0.349 * r + 0.686 * g + 0.168 * b
        tb = 0.272 * r + 0.534 * g + 0.131 * b
        r, g, b = tr, tg, tb
    elif scheme == "cyan_magenta":
        b = np.clip(b * 1.4 + g * 0.2, 0, 255)
        r = np.clip(r * 1.3 + b * 0.15, 0, 255)
        g = np.clip(g * 0.7, 0, 255)
    elif scheme == "blood":
        r = np.clip(r * 1.5 + 30, 0, 255)
        g = np.clip(g * 0.5, 0, 255)
        b = np.clip(b * 0.4, 0, 255)
    else:
        return frame
    tinted = cv2.merge([np.clip(b, 0, 255), np.clip(g, 0, 255), np.clip(r, 0, 255)])
    return np.where(mask[:, :, np.newaxis] == 255, frame, tinted.astype(np.uint8))


def create_face_safe_mask(h: int, w: int) -> np.ndarray:
    """
    Central ellipse protecting eyes/nose/teeth silhouette.
    Allow: background + edges + shoulders. Keeps PFP readable.
    """
    cx, cy = w / 2, h / 2
    rx, ry = w * 0.35, h * 0.4  # central face region
    y = np.arange(h, dtype=np.float32)[:, np.newaxis]
    x = np.arange(w, dtype=np.float32)[np.newaxis, :]
    ellipse = ((x - cx) ** 2 / rx**2 + (y - cy) ** 2 / ry**2) <= 1.0
    return (ellipse.astype(np.uint8) * 255).squeeze()


def create_auto_mask(
    frame: np.ndarray,
    use_background_subtraction: bool = True,
    sensitivity: float = 0.5,
    face_safe: bool = True,
) -> np.ndarray:
    """
    Auto-detect foreground (to protect) using edge detection.
    face_safe: also protect central face region (eyes/nose/teeth) for PFP readability.
    Returns mask where 255 = protect (foreground), 0 = glitch (background).
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sens = np.clip(sensitivity, 0.01, 0.99)

    low = int(30 + (1 - sens) * 80)
    high = int(80 + sens * 120)
    edges = cv2.Canny(gray, low, high)

    dilate_size = int(8 + sens * 25)
    dilate_size = max(5, dilate_size)
    kernel = np.ones((dilate_size, dilate_size), np.uint8)
    edge_mask = cv2.dilate(edges, kernel)

    if use_background_subtraction:
        kernel_small = np.ones((5, 5), np.uint8)
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_small)
        thresh = int(20 + sens * 30)
        _, gradient_thresh = cv2.threshold(gradient, thresh, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_or(edge_mask, gradient_thresh)
    else:
        combined = edge_mask

    fill_size = int(15 + sens * 30)
    combined = cv2.dilate(combined, np.ones((fill_size, fill_size), np.uint8))
    combined = cv2.erode(combined, np.ones((8, 8), np.uint8))

    if face_safe:
        face_mask = create_face_safe_mask(h, w)
        combined = cv2.bitwise_or(combined, face_mask)

    return combined


def make_soft_mask(mask_binary: np.ndarray, soft_pixels: int) -> np.ndarray:
    """Create feathered mask. soft_pixels=0 returns binary; higher = softer transition."""
    if soft_pixels <= 0:
        return (mask_binary > 127).astype(np.float32)
    k = max(1, soft_pixels * 2 + 1)
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(mask_binary.astype(np.float32), (k, k), soft_pixels * 0.5)
    return np.clip(blurred / 255.0, 0, 1).astype(np.float32)


def apply_foreground_effects(frame: np.ndarray, mask: np.ndarray, intensity: int, frame_idx: int) -> np.ndarray:
    """Subtle effects applied only to the masked (foreground) region."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    b, g, r = cv2.split(frame.astype(np.float32))
    np.random.seed(frame_idx * 1111)
    pulse = 0.7 + 0.3 * np.sin(frame_idx * 0.4)
    amt = intensity * 0.03 * pulse
    b_new = np.clip(b + (r - b) * amt, 0, 255).astype(np.float32)
    r_new = np.clip(r + (b - r) * amt * 0.5, 0, 255).astype(np.float32)
    shifted = np.stack([b_new, g, r_new], axis=-1).astype(np.uint8)
    grain = np.random.randint(-int(intensity * 4), int(intensity * 4), (h, w, 3), dtype=np.int32)
    shifted = np.clip(shifted.astype(np.int32) + grain, 0, 255).astype(np.uint8)
    mask_u8 = (mask > 0.5).astype(np.uint8)
    return np.where(mask_u8[:, :, np.newaxis] == 1, shifted, frame).astype(np.uint8)


def apply_subject_invert(frame: np.ndarray, mask: np.ndarray, intensity: int, frame_idx: int) -> np.ndarray:
    """Invert colors on the subject - intensity controls blend amount (partial to full invert)."""
    if intensity <= 0:
        return frame
    inv = 255 - frame
    amt = min(1.0, intensity * 0.1)
    pulse = 0.6 + 0.4 * np.sin(frame_idx * 0.5)
    amt = amt * pulse
    mask_3ch = (mask[:, :, np.newaxis] > 0.5).astype(np.float32)
    blended = (1 - amt) * frame.astype(np.float32) + amt * inv.astype(np.float32)
    result = mask_3ch * blended + (1 - mask_3ch) * frame.astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_subject_particles(frame: np.ndarray, mask: np.ndarray, intensity: int, frame_idx: int) -> np.ndarray:
    """Grainy particle overlay on subject - white and dark specks that animate per frame."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    np.random.seed(frame_idx * 7777 + 3333)
    n_particles = int(h * w * 0.0015 * intensity * (0.5 + np.random.rand()))
    n_particles = max(200, min(8000, n_particles))
    ys = np.random.randint(0, h, n_particles)
    xs = np.random.randint(0, w, n_particles)
    vals = np.random.choice([0, 255], n_particles, p=[0.5, 0.5]).astype(np.float32)
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[ys, xs, :] = vals[:, np.newaxis]
    subject = mask > 0.3
    hit = np.zeros((h, w), dtype=bool)
    hit[ys, xs] = True
    apply_mask = hit & subject
    r = min(1.0, intensity * 0.12)
    result = frame.astype(np.float32).copy()
    result[apply_mask] = (1 - r) * result[apply_mask] + r * overlay[apply_mask]
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_chroma_dropout(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """YUV: drop chroma (grey-ish) for 1–2 frames, then oversaturate rebound. VHS authenticity."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    np.random.seed(frame_idx * 7777)
    # Cycle: dropout (grey) vs rebound (oversaturated)
    phase = (frame_idx // 2) % 4  # 2 frames dropout, 2 frames rebound
    if phase in (0, 1):
        # Dropout: reduce chroma (U,V in YUV) -> grey
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y_ch, u_ch, v_ch = cv2.split(yuv)
        drop = 1.0 - min(0.9, intensity * 0.12 + chaos * 0.2)
        u_new = np.clip(128 + (u_ch.astype(np.float32) - 128) * drop, 0, 255).astype(np.uint8)
        v_new = np.clip(128 + (v_ch.astype(np.float32) - 128) * drop, 0, 255).astype(np.uint8)
        yuv = cv2.merge([y_ch, u_new, v_new])
        glitched = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        # Rebound: oversaturate
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_h, s_h, v_h = cv2.split(hsv)
        boost = 1.0 + min(1.5, intensity * 0.15 + chaos * 0.3)
        s_new = np.clip(s_h.astype(np.float32) * boost, 0, 255).astype(np.uint8)
        hsv = cv2.merge([h_h, s_new, v_h])
        glitched = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def apply_flow_warp(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Displacement field warp - liquid space-time distortion. Rare: 1-2 bands. Legendary: warp from skull."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    np.random.seed(frame_idx * 11919)
    res = 64
    phase = frame_idx * 0.15
    yy = np.linspace(0, h, res)
    xx = np.linspace(0, w, res)
    cx, cy = w / 2, h / 2
    dist = np.sqrt((np.outer(np.ones(res), xx) - cx) ** 2 + (np.outer(yy, np.ones(res)) - cy) ** 2)
    r_norm = dist / max(cx, cy)
    amp = (2 + intensity * 1.5) * (0.6 + chaos * 0.6)
    wave_x = np.sin(r_norm * 4 + phase) * amp
    wave_y = np.cos(r_norm * 3 + phase * 1.2) * amp * 0.6
    map_x_low = np.tile(np.arange(res, dtype=np.float32) * (w / res), (res, 1)) + wave_x
    map_y_low = np.tile(np.arange(res, dtype=np.float32)[:, np.newaxis] * (h / res), (1, res)) + wave_y
    map_x = cv2.resize(map_x_low, (w, h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y_low, (w, h), interpolation=cv2.INTER_LINEAR)
    glitch_region = (mask < 128).astype(np.float32)
    xc = np.arange(w, dtype=np.float32)[np.newaxis, :]
    yc = np.arange(h, dtype=np.float32)[:, np.newaxis]
    map_x = np.where(glitch_region > 0.5, map_x, xc).astype(np.float32)
    map_y = np.where(glitch_region > 0.5, map_y, yc).astype(np.float32)
    glitched = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def apply_slit_scan(
    frame: np.ndarray, prev_frame: Optional[np.ndarray], mask: np.ndarray,
    intensity: int, chaos: float, frame_idx: int
) -> np.ndarray:
    """Slit-scan / time-smear: vertical slices from past frames stitch into current. Temporal ripping."""
    if prev_frame is None or intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    pos = frame_idx % EVENT_CYCLE
    if pos not in (EVENT_PEAK_FRAME - 1, EVENT_PEAK_FRAME, EVENT_PEAK_FRAME + 1):
        return frame
    np.random.seed(frame_idx * 3313)
    slice_w = max(2, 8 - intensity)
    glitched = frame.copy().astype(np.float32)
    n_slices = w // slice_w
    for i in range(n_slices):
        x = i * slice_w
        if x + slice_w > w:
            break
        if mask[:, x:x + slice_w].mean() > 220:
            continue
        t = np.random.rand()
        blend = t * frame[:, x:x + slice_w].astype(np.float32) + (1 - t) * prev_frame[:, x:x + slice_w].astype(np.float32)
        glitched[:, x:x + slice_w] = blend
    return np.where(mask[:, :, np.newaxis] == 255, frame, np.clip(glitched, 0, 255).astype(np.uint8)).astype(np.uint8)


def apply_parallax_split(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Parallax: depth layers (bg / edge / face) offset differently - soul leaving body."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    pos = frame_idx % EVENT_CYCLE
    if pos not in (EVENT_PEAK_FRAME - 1, EVENT_PEAK_FRAME):
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    edge_dilate = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    bg_mask = (mask < 64).astype(np.uint8) * 255
    edge_mask = (edge_dilate > 0) & (mask < 200)
    face_mask = (mask > 200).astype(np.uint8) * 255
    offset_x = int(4 * intensity * (0.5 + 0.5 * np.sin(frame_idx)))
    offset_y = int(-2 * intensity)
    M_bg = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    M_edge = np.float32([[1, 0, -offset_x // 2], [0, 1, offset_y // 2]])
    bg_shifted = cv2.warpAffine(frame, M_bg, (w, h), borderMode=cv2.BORDER_REPLICATE)
    edge_shifted = cv2.warpAffine(frame, M_edge, (w, h), borderMode=cv2.BORDER_REPLICATE)
    glitched = np.where(face_mask[:, :, np.newaxis] == 255, frame, bg_shifted)
    glitched = np.where(edge_mask[:, :, np.newaxis], edge_shifted, glitched)
    return np.clip(glitched, 0, 255).astype(np.uint8)


def apply_voronoi_shatter(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Tessellated Voronoi shatter - crystalline fracture. Rare: small region. Legendary: full-frame face-safe."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    np.random.seed(frame_idx * 7171)
    n_points = max(8, 24 - intensity * 2) if intensity < 7 else max(20, 40 - intensity)
    pts = np.random.randint(0, max(1, min(h, w)), (n_points, 2))
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
    offset_scale = intensity * (0.8 + chaos * 0.6)
    glitched = frame.copy().astype(np.float32)
    for i in range(n_points):
        cell = (id_map == i) & (mask < 128)
        if not cell.any():
            continue
        ox = int(np.clip(np.random.randn() * offset_scale, -8, 8))
        oy = int(np.clip(np.random.randn() * offset_scale * 0.5, -4, 4))
        if ox != 0 or oy != 0:
            shifted = np.roll(np.roll(frame, ox, axis=1), oy, axis=0)
            glitched[cell] = shifted[cell]
    return np.where(mask[:, :, np.newaxis] == 255, frame, np.clip(glitched, 0, 255).astype(np.uint8)).astype(np.uint8)


def apply_echo_crown(
    frame: np.ndarray, prev_frame: Optional[np.ndarray], mask: np.ndarray,
    intensity: int, frame_idx: int
) -> np.ndarray:
    """Directional echoes with decay - crown/halo behind skull. Legendary tier."""
    if prev_frame is None or intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    pos = frame_idx % EVENT_CYCLE
    decay = 1.0 if 5 <= pos <= 7 else (0.4 if 4 <= pos <= 8 else 0.15)
    amt = min(0.55, intensity * 0.08 * decay)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 120)
    edge_weight = cv2.dilate(edges, np.ones((2, 2), np.uint8)).astype(np.float32) / 255.0
    edge_weight = edge_weight * (1 - mask.astype(np.float32) / 255.0)
    blend = cv2.addWeighted(frame, 1 - amt, prev_frame, amt, 0)
    result = (1 - edge_weight[:, :, np.newaxis]) * frame.astype(np.float32) + edge_weight[:, :, np.newaxis] * blend.astype(np.float32)
    return np.where(mask[:, :, np.newaxis] == 255, frame, np.clip(result, 0, 255).astype(np.uint8)).astype(np.uint8)


def apply_strobe_phase(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Strobe: lock luminance to few levels (posterize) for 1 boundary frame - reality sampling."""
    if intensity <= 0:
        return frame
    pos = frame_idx % EVENT_CYCLE
    if pos != EVENT_PEAK_FRAME:
        return frame
    levels = max(4, 12 - intensity)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    posterized = (gray // (256 // levels)) * (256 // levels)
    posterized = np.clip(posterized, 0, 255).astype(np.uint8)
    ratio = (posterized.astype(np.float32) + 1) / (gray.astype(np.float32) + 1)
    glitched = np.clip(frame.astype(np.float32) * ratio[:, :, np.newaxis], 0, 255).astype(np.uint8)
    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def apply_frame_drop(
    frame: np.ndarray, prev_frame: Optional[np.ndarray], mask: np.ndarray,
    intensity: int, chaos: float, frame_idx: int
) -> np.ndarray:
    """Frame drop: repeat prev frame, then strong hit on next. VHS-authentic."""
    if prev_frame is None or intensity <= 0:
        return frame
    pos = frame_idx % EVENT_CYCLE
    if pos == EVENT_PEAK_FRAME - 1:
        return prev_frame
    if pos == EVENT_PEAK_FRAME:
        hit = frame.copy().astype(np.float32)
        b, g, r = cv2.split(frame)
        shift = max(1, intensity)
        hit = cv2.merge([np.roll(b, -shift, axis=1), g, np.roll(r, shift, axis=1)])
        return np.where(mask[:, :, np.newaxis] == 255, frame, np.clip(hit, 0, 255).astype(np.uint8)).astype(np.uint8)
    return frame


def apply_palette_cycle(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Palette cycling - shift hue on background. Pixel-authentic, compresses well."""
    if intensity <= 0:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_h, s_h, v_h = cv2.split(hsv)
    cycle = (frame_idx * 3) % 180
    h_shifted = ((h_h.astype(np.int32) + cycle) % 180).astype(np.uint8)
    hsv = cv2.merge([h_shifted, s_h, v_h])
    glitched = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    amt = min(0.7, intensity * 0.12)
    blended = (1 - amt) * frame.astype(np.float32) + amt * glitched.astype(np.float32)
    return np.where(mask[:, :, np.newaxis] == 255, frame, np.clip(blended, 0, 255).astype(np.uint8)).astype(np.uint8)


def apply_chroma_collapse(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Selective color channel collapse: near-monochrome then oversaturate rebound - soul drain."""
    if intensity <= 0:
        return frame
    pos = (frame_idx // 2) % 4
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y_ch, u_ch, v_ch = cv2.split(yuv)
    if pos in (0, 1):
        drop = 1.0 - min(0.85, intensity * 0.1)
        u_new = np.clip(128 + (u_ch.astype(np.float32) - 128) * drop, 0, 255).astype(np.uint8)
        v_new = np.clip(128 + (v_ch.astype(np.float32) - 128) * drop, 0, 255).astype(np.uint8)
        yuv = cv2.merge([y_ch, u_new, v_new])
    else:
        boost = 1.0 + min(1.8, intensity * 0.2)
        u_new = np.clip(128 + (u_ch.astype(np.float32) - 128) * boost, 0, 255).astype(np.uint8)
        v_new = np.clip(128 + (v_ch.astype(np.float32) - 128) * boost, 0, 255).astype(np.uint8)
        yuv = cv2.merge([y_ch, u_new, v_new])
    glitched = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def apply_ordered_dither(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Bayer ordered dither overlay - retro-printed / ritual-encoded."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    bayer4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float32) / 16.0
    bayer = np.tile(bayer4, (h // 4 + 2, w // 4 + 2))[:h, :w]
    thresh = bayer * (0.3 + intensity * 0.08)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    dithered = (gray > thresh).astype(np.uint8) * 255
    dithered_3ch = np.stack([dithered, dithered, dithered], axis=-1)
    amt = min(0.35, intensity * 0.06)
    glitched = (1 - amt) * frame.astype(np.float32) + amt * dithered_3ch.astype(np.float32)
    return np.where(mask[:, :, np.newaxis] == 255, frame, np.clip(glitched, 0, 255).astype(np.uint8)).astype(np.uint8)


def apply_sigil_ring(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Sigil ring: broken ring + tick marks + gaps + pulse. No runes, iconic LICHE."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    cx, cy = w / 2, h / 2
    y = np.arange(h, dtype=np.float32)[:, np.newaxis] - cy
    x = np.arange(w, dtype=np.float32)[np.newaxis, :] - cx
    r = np.sqrt(x**2 + y**2)
    np.random.seed(frame_idx * 4444)
    pulse = 0.5 + 0.5 * np.sin(frame_idx * 0.5)
    ring_r = min(h, w) * (0.35 + 0.1 * pulse)
    ring_w = max(1, 2 + intensity // 2)
    ring_mask = (np.abs(r - ring_r) < ring_w).astype(np.float32)
    gaps = (np.arange(w) + frame_idx * 3) % 17 > 12
    ring_mask[:, gaps] = 0
    tick_step = 8 + (frame_idx % 4)
    for a in range(0, 360, tick_step):
        rad = np.deg2rad(a)
        tx = int(cx + ring_r * np.cos(rad))
        ty = int(cy + ring_r * np.sin(rad))
        if 2 <= tx < w - 2 and 2 <= ty < h - 2 and mask[ty, tx] < 200:
            ring_mask[ty - 1:ty + 2, tx - 1:tx + 2] = 1.0
    color = np.array([30, 180, 180], dtype=np.float32)
    glitched = frame.astype(np.float32) + ring_mask[:, :, np.newaxis] * color * (0.15 + intensity * 0.03)
    return np.where(mask[:, :, np.newaxis] == 255, frame, np.clip(glitched, 0, 255).astype(np.uint8)).astype(np.uint8)


def apply_phylactery_glow(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Phylactery: floating orb behind skull, pulses, distorts field (warp + chroma bleed)."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    cx, cy = w / 2, h * 0.4
    ox = cx + int(15 * np.sin(frame_idx * 0.3))
    oy = int(cy + 8 * np.cos(frame_idx * 0.25))
    y = np.arange(h, dtype=np.float32)[:, np.newaxis] - oy
    x = np.arange(w, dtype=np.float32)[np.newaxis, :] - ox
    r = np.sqrt(x**2 + y**2)
    orb_r = 8 + intensity
    orb_mask = np.exp(-r**2 / (2 * (orb_r ** 2))).astype(np.float32)
    orb_mask = orb_mask * (1 - mask.astype(np.float32) / 255.0)
    pulse = 0.7 + 0.3 * np.sin(frame_idx * 0.6)
    glow_color = np.array([180, 255, 255], dtype=np.float32) * pulse
    glitched = frame.astype(np.float32) + orb_mask[:, :, np.newaxis] * glow_color * (intensity * 0.04)
    return np.clip(glitched, 0, 255).astype(np.uint8)


def apply_abyss_window(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Abyss window: cut band, fill with void/starfield, rim glow. Portal interior."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    pos = frame_idx % EVENT_CYCLE
    if pos not in (EVENT_PEAK_FRAME - 1, EVENT_PEAK_FRAME):
        return frame
    np.random.seed(frame_idx * 5555)
    band_h = max(8, int(h * 0.08 * intensity))
    y0 = int(h * (0.2 + 0.3 * np.sin(frame_idx * 0.2)))
    y0 = max(0, min(y0, h - band_h - 1))
    y1 = y0 + band_h
    if mask[y0:y1, :].mean() > 180:
        return frame
    void = np.zeros((band_h, w, 3), dtype=np.uint8)
    stars = np.random.rand(band_h, w) < 0.02
    void[stars] = [255, 255, 255]
    rim = np.zeros((band_h, w, 3), dtype=np.float32)
    rim[0, :] = [60, 80, 60]
    rim[-1, :] = [60, 80, 60]
    rim[:, 0] = [60, 80, 60]
    rim[:, -1] = [60, 80, 60]
    glitched = frame.copy()
    glitched[y0:y1, :] = np.clip(void.astype(np.float32) + rim, 0, 255).astype(np.uint8)
    return glitched


def apply_fft_jitter(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """FFT phase jitter - subtle surreal shimmer. Face-protected."""
    if intensity <= 0:
        return frame
    pos = frame_idx % EVENT_CYCLE
    if pos != EVENT_PEAK_FRAME:
        return frame
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fft = np.fft.fft2(gray)
    np.random.seed(frame_idx * 6666)
    phase_jitter = np.exp(1j * np.random.randn(h, w) * (intensity * 0.05))
    fft_j = fft * phase_jitter
    shimmer = np.real(np.fft.ifft2(fft_j)).astype(np.float32)
    ratio = (shimmer + 128) / (gray + 128)
    glitched = np.clip(frame.astype(np.float32) * ratio[:, :, np.newaxis], 0, 255).astype(np.uint8)
    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def apply_moire_grid(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Moire: angled grid drifts 1px/frame, interacts with content. Cursed signal."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    drift = (frame_idx % 4) - 2
    line_int = max(4, 12 - intensity)
    grid = np.zeros((h, w), dtype=np.float32)
    angle = 0.3 + 0.1 * np.sin(frame_idx * 0.1)
    for i in range(-20, 25):
        x0 = int(w / 2 + i * 30 * np.cos(angle) + drift) % (w + 50) - 25
        for y in range(h):
            x = int(x0 + y * np.tan(angle))
            if 0 <= x < w and mask[y, x] < 200:
                grid[y, max(0, x - 1):min(w, x + 2)] = 0.15
    glitched = frame.astype(np.float32) * (1 - grid[:, :, np.newaxis]) + grid[:, :, np.newaxis] * 128
    return np.where(mask[:, :, np.newaxis] == 255, frame, np.clip(glitched, 0, 255).astype(np.uint8)).astype(np.uint8)


def apply_ghost_trail(
    frame: np.ndarray, prev_frame: Optional[np.ndarray], mask: np.ndarray,
    intensity: int, frame_idx: int
) -> np.ndarray:
    """Echo/ghost blend with previous frame. Trail decays fast - don't smear skull for 12 frames."""
    if prev_frame is None or intensity <= 0:
        return frame
    # Fast decay: only strong near event peak (frames 5–7), else minimal
    pos = frame_idx % EVENT_CYCLE
    decay = 1.0 if 4 <= pos <= 8 else (0.3 if 3 <= pos <= 9 else 0.1)
    amt = min(0.5, intensity * 0.06 * decay)
    blend = cv2.addWeighted(frame, 1 - amt, prev_frame, amt, 0)
    return np.where(mask[:, :, np.newaxis] == 255, frame, blend).astype(np.uint8)


def apply_block_tear(frame: np.ndarray, mask: np.ndarray, intensity: int, chaos: float, frame_idx: int) -> np.ndarray:
    """Block tear with snap-back (repair moment) - feels possessed. Coherent noise for tear positions."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    np.random.seed(frame_idx * 4321)
    slice_h = max(6, 24 - intensity * 2)
    glitched = frame.copy()
    max_slices = max(2, 8 - intensity)
    # Snap-back: repair moment (no tear) near event decay
    pos = frame_idx % EVENT_CYCLE
    repair = 7 <= pos <= 10 and np.random.rand() < 0.4
    if repair:
        return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)

    # Coherent noise: tear positions tied to frame_idx so they track consistently
    base_y = (frame_idx * 17) % max(1, h - slice_h * max_slices)
    n_slices = 0
    for i in range(max_slices):
        if n_slices >= max_slices:
            break
        y = (base_y + i * (h // max_slices)) % max(1, h - slice_h)
        if y + slice_h > h:
            continue
        if mask[y:y+slice_h, :].mean() < 220:
            offset = int(np.random.randn() * intensity * 4)
            offset = np.clip(offset, -intensity * 6, intensity * 6)
            if offset != 0:
                glitched[y:y+slice_h, :] = np.roll(frame[y:y+slice_h, :].copy(), offset, axis=1)
                n_slices += 1
    return np.where(mask[:, :, np.newaxis] == 255, frame, glitched).astype(np.uint8)


def apply_transition_hit(
    frame: np.ndarray,
    mask: np.ndarray,
    strength: float,
    frame_idx: int,
    token_seed: int,
    preset_name: str,
    prev_frame: Optional[np.ndarray] = None,
    hit_components: tuple = ("block_tear", "rgb_shift"),
) -> np.ndarray:
    """
    Localized transition hit at A/B swap boundary. Band-limited, pixel-safe.
    strength: 0..1. hit_components: subset of effects to apply.
    """
    if strength <= 0:
        return frame
    h, w = frame.shape[:2]
    glitch_region = (mask < 128)
    seed_val = int((token_seed * 0xAB155A1 + abs(hash(preset_name)) % (2**32) + frame_idx * 7919) % (2**32))
    np.random.seed(seed_val)
    random.seed(seed_val)

    # Band zone: 1-2 horizontal bands or diagonal rip (deterministic from frame_idx)
    use_diagonal = (frame_idx // 6) % 2 == 0
    band_h = max(8, int(h * 0.12 * strength))
    band_h = min(band_h, h // 4)

    if use_diagonal:
        slope = 0.2 + 0.2 * np.sin(frame_idx * 0.1)
        y_center = h * (0.35 + 0.3 * np.sin(frame_idx * 0.15))
        x = np.arange(w, dtype=np.float32)
        y_line = slope * (x - w / 2) + y_center
        dist = np.abs(np.arange(h, dtype=np.float32)[:, np.newaxis] - y_line[np.newaxis, :])
        band_mask = (dist < band_h).astype(np.float32)
    else:
        y_center = int(h * (0.3 + 0.4 * (frame_idx % 3) / 3))
        y0 = max(0, y_center - band_h // 2)
        y1 = min(h, y_center + band_h // 2)
        band_mask = np.zeros((h, w), dtype=np.float32)
        band_mask[y0:y1, :] = 1.0

    apply_mask = (band_mask > 0.01) & glitch_region
    if not apply_mask.any():
        return frame

    glitched = frame.copy().astype(np.float32)
    eff_strength = strength * 0.8

    if "block_tear" in hit_components:
        slice_h = max(4, band_h // 2)
        for y in range(0, h, slice_h):
            if y + slice_h > h:
                break
            strip = apply_mask[y:y + slice_h, :]
            if strip.any():
                offset = int(np.clip(np.random.randn() * eff_strength * 8, -12, 12))
                if offset != 0:
                    glitched[y:y + slice_h, :] = np.roll(glitched[y:y + slice_h, :].copy(), offset, axis=1)
                    glitched[y:y + slice_h, :] = np.where(
                        mask[y:y + slice_h, :, np.newaxis] == 255,
                        frame[y:y + slice_h, :].astype(np.float32),
                        glitched[y:y + slice_h, :],
                    )

    if "rgb_shift" in hit_components:
        shift = max(1, int(3 * eff_strength))
        b, g, r = cv2.split(glitched.astype(np.uint8))
        r_shifted = np.roll(r, shift, axis=1)
        b_shifted = np.roll(b, -shift, axis=1)
        shifted = cv2.merge([b_shifted, g, r_shifted]).astype(np.float32)
        alpha = band_mask * (glitch_region.astype(np.float32)) * eff_strength * 0.6
        glitched = (1 - alpha[:, :, np.newaxis]) * glitched + alpha[:, :, np.newaxis] * shifted

    if "vhs_tracking" in hit_components:
        band_height = max(2, int(12 * eff_strength))
        for y in range(0, h, band_height):
            strip = apply_mask[y:min(y + band_height, h), :]
            if strip.any() and mask[y:min(y + band_height, h), :].mean() < 200:
                offset = int(np.clip(np.random.randn() * eff_strength * 6, -10, 10))
                if offset != 0:
                    sy, ey = y, min(y + band_height, h)
                    glitched[sy:ey, :] = np.roll(glitched[sy:ey, :].copy(), offset, axis=1)
                    glitched[sy:ey, :] = np.where(
                        mask[sy:ey, :, np.newaxis] == 255,
                        frame[sy:ey, :].astype(np.float32),
                        glitched[sy:ey, :],
                    )

    if "rolling_scanlines" in hit_components:
        line_interval = max(2, 6 - int(eff_strength * 4))
        dark_val = int(255 * (1 - eff_strength * 0.15))
        for i in range(0, h, line_interval):
            line_y = (i + frame_idx) % h
            if (band_mask[line_y, :] > 0.1).any() and mask[line_y, :].mean() < 220:
                glitched[line_y, :] = (0.7 * glitched[line_y, :] + 0.3 * dark_val).astype(np.float32)

    if "brightness_pulse" in hit_components:
        pulse = 1.0 + eff_strength * 0.15 * (0.5 + 0.5 * np.sin(frame_idx * 2))
        glitched[apply_mask] = np.clip(glitched[apply_mask] * pulse, 0, 255)

    if "chromatic_aberration" in hit_components:
        shift = max(1, int(4 * eff_strength))
        b, g, r = cv2.split(glitched.astype(np.uint8))
        r_s = np.roll(r, shift, axis=1)
        b_s = np.roll(b, -shift, axis=1)
        shifted = cv2.merge([b_s, g, r_s]).astype(np.float32)
        alpha = band_mask * (glitch_region.astype(np.float32)) * eff_strength * 0.5
        glitched = (1 - alpha[:, :, np.newaxis]) * glitched + alpha[:, :, np.newaxis] * shifted

    if "abyssal_tear" in hit_components:
        void_bgr = np.array([0, 0, 0], dtype=np.float32)
        rim_cyan = np.array([60, 60, 20], dtype=np.float32)
        rim_magenta = np.array([60, 20, 60], dtype=np.float32)
        rim_px = 2
        band_u8 = (band_mask > 0.5).astype(np.uint8) * 255
        band_eroded = cv2.erode(band_u8, np.ones((rim_px * 2 + 1, rim_px * 2 + 1), np.uint8))
        rim_u8 = band_u8 - band_eroded
        inner_mask = (band_eroded > 0).astype(np.float32)[:, :, np.newaxis]
        rim_mask = (rim_u8 > 0).astype(np.float32)[:, :, np.newaxis]
        tear_fill = inner_mask * void_bgr + rim_mask * (rim_cyan if frame_idx % 2 == 0 else rim_magenta)
        tear_fill = np.clip(tear_fill, 0, 255)
        hit_alpha = band_mask[:, :, np.newaxis] * (glitch_region.astype(np.float32)[:, :, np.newaxis]) * eff_strength * 0.9
        glitched = (1 - hit_alpha) * glitched + hit_alpha * tear_fill

    if "ghost_trail" in hit_components and prev_frame is not None:
        amt = min(0.4, eff_strength * 0.5)
        ghost_blend = cv2.addWeighted(glitched.astype(np.uint8), 1 - amt, prev_frame, amt, 0)
        alpha = band_mask[:, :, np.newaxis] * (glitch_region.astype(np.float32)[:, :, np.newaxis]) * 0.7
        glitched = (1 - alpha) * glitched + alpha * ghost_blend.astype(np.float32)

    result = np.clip(glitched, 0, 255).astype(np.uint8)
    return np.where(mask[:, :, np.newaxis] == 255, frame, result).astype(np.uint8)


def apply_vignette(frame: np.ndarray, mask: np.ndarray, intensity: int, frame_idx: int) -> np.ndarray:
    """Final grade - subtle vignette. Too strong makes everything muddy."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    cx, cy = w / 2, h / 2
    x = np.arange(w, dtype=np.float32) - cx
    y = np.arange(h, dtype=np.float32) - cy
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2) / max(cx, cy)
    # Softer gradient: power curve keeps center bright longer, gentler falloff
    r_soft = r ** 0.6
    fade = np.clip(1 - r_soft * (0.25 + intensity * 0.02), 0, 1)
    fade_3ch = np.stack([fade, fade, fade], axis=-1)
    return (frame.astype(np.float32) * fade_3ch).astype(np.uint8)


def apply_bloom(frame: np.ndarray, mask: np.ndarray, intensity: int, frame_idx: int) -> np.ndarray:
    """Soft glow on very limited emissive pixels (eyes/runes) - don't touch outlines."""
    if intensity <= 0:
        return frame
    # Restrict to very bright pixels only (emissive) - avoid smudging pixel art
    bright = cv2.inRange(frame, (235, 235, 235), (255, 255, 255))
    bright = cv2.bitwise_and(bright, mask)
    # Erode to keep only small bright spots, not large areas
    kernel = np.ones((2, 2), np.uint8)
    bright = cv2.erode(bright, kernel)
    blur_size = min(7, max(3, intensity * 2) | 1)
    glow = cv2.GaussianBlur(cv2.merge([bright, bright, bright]), (blur_size, blur_size), 0)
    amt = intensity * 0.03
    result = np.clip(frame.astype(np.float32) + glow.astype(np.float32) * amt, 0, 255).astype(np.uint8)
    return np.where(mask[:, :, np.newaxis] == 255, result, frame).astype(np.uint8)


def apply_edge_pulse(
    frame: np.ndarray, mask_soft: np.ndarray, intensity: int, frame_idx: int
) -> np.ndarray:
    """Chromatic effect on the edge band that pulses in/out over time."""
    if intensity <= 0:
        return frame
    pulse = 0.2 + 0.8 * (0.5 + 0.5 * np.sin(frame_idx * 0.6))
    amt = min(1.0, intensity * pulse * 0.12)
    edge_band = np.abs(mask_soft - 0.5) * 2
    edge_band = np.clip(1 - edge_band, 0, 1)
    edge_weight = np.clip(edge_band * amt, 0, 1).astype(np.float32)
    b, g, r = cv2.split(frame.astype(np.float32))
    offset = int(3 * amt)
    r_shifted = np.roll(r, offset, axis=1)
    b_shifted = np.roll(b, -offset, axis=1)
    edge_effect = cv2.merge([b_shifted, g, r_shifted])
    blend = (
        (1 - edge_weight[:, :, np.newaxis]) * frame.astype(np.float32)
        + edge_weight[:, :, np.newaxis] * edge_effect
    )
    return np.clip(blend, 0, 255).astype(np.uint8)


def _all_effects_disabled(params: GlitchParams) -> bool:
    """Strong safety check — if literally zero effects are active, return original frame unchanged."""
    effect_list = [
        params.rgb_shift_intensity,
        params.chromatic_aberration,
        params.scanlines,
        params.digital_noise,
        params.pixelation,
        params.datamosh,
        params.melting,
        params.vhs_tracking,
        params.crt_flicker,
        params.bitcrush,
        params.tv_static,
        params.frame_drift,
        params.rolling_scanlines,
        params.vignette,
        params.bloom,
        params.film_grain,
        params.ghost_trail,
        params.block_tear,
        params.pixel_sort,
        params.neon_bars,
        params.edge_dissolve,
        params.chroma_dropout,
        params.flow_warp,
        params.slit_scan,
        params.parallax_split,
        params.voronoi_shatter,
        params.echo_crown,
        params.strobe_phase,
        params.frame_drop,
        params.palette_cycle,
        params.chroma_collapse,
        params.ordered_dither,
        params.sigil_ring,
        params.phylactery_glow,
        params.abyss_window,
        params.fft_jitter,
        params.moire_grid,
        params.foreground_effect,
        params.subject_invert,
        params.subject_particles,
        params.edge_pulse,
        params.shader_necrotic_iridescent_flow_intensity,
        params.shader_hexagonal_warp_intensity,
        params.shader_caustic_flow_intensity,
        params.shader_thermal_distort_intensity,
    ]
    if any(v > 0 for v in effect_list) or params.color_scheme not in (None, "default", "custom"):
        return False
    return True


def process_frame(frame: np.ndarray, mask: np.ndarray, params: GlitchParams,
                  frame_idx: int, prev_frame: Optional[np.ndarray],
                  progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
    """
    NUCLEAR BULLETPROOF VERSION — forces clean original frame if literally zero effects.
    This fixes the persistent inversion bug once and for all.
    """
    # === ABSOLUTE ZERO-EFFECTS CHECK (catches everything) ===
    if (params.rgb_shift_intensity == 0 and params.chromatic_aberration == 0 and params.scanlines == 0 and
        params.digital_noise == 0 and params.pixelation == 0 and params.datamosh == 0 and params.melting == 0 and
        params.vhs_tracking == 0 and params.crt_flicker == 0 and params.bitcrush == 0 and params.tv_static == 0 and
        params.frame_drift == 0 and params.rolling_scanlines == 0 and params.ghost_trail == 0 and
        params.block_tear == 0 and params.vignette == 0 and params.bloom == 0 and params.film_grain == 0 and
        params.pixel_sort == 0 and params.neon_bars == 0 and params.edge_dissolve == 0 and
        params.chroma_dropout == 0 and params.flow_warp == 0 and params.slit_scan == 0 and
        params.parallax_split == 0 and params.voronoi_shatter == 0 and params.echo_crown == 0 and
        params.strobe_phase == 0 and params.frame_drop == 0 and params.palette_cycle == 0 and
        params.chroma_collapse == 0 and params.ordered_dither == 0 and params.sigil_ring == 0 and
        params.phylactery_glow == 0 and params.abyss_window == 0 and params.fft_jitter == 0 and
        params.moire_grid == 0 and         params.foreground_effect == 0 and params.subject_invert == 0 and
        params.subject_particles == 0 and params.edge_pulse == 0 and
        params.shader_necrotic_iridescent_flow_intensity == 0 and
        params.shader_hexagonal_warp_intensity == 0 and
        params.shader_caustic_flow_intensity == 0 and
        params.shader_thermal_distort_intensity == 0 and
        params.color_scheme in (None, "default", "custom", "")):

        print(f"[Liche DEBUG] Frame {frame_idx} — ZERO effects active. Returning CLEAN original frame.")
        return frame.copy()

    # Extra safety — never allow inversion when subject_invert is off
    if params.subject_invert == 0 and params.color_scheme == "inverted":
        params = GlitchParams(**{**vars(params), "color_scheme": "default"})

    # === Normal processing continues only if effects are active ===
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_soft = make_soft_mask(mask_binary, params.mask_soft_edge)
    mask_zeros = np.zeros_like(mask_binary)

    # Effects that displace/warp pixels: mask-protected (subject stays clean)
    # Effects that are overlay-only (scanlines, grain, etc.): mask-exempt (can affect subject)
    bg_result = frame.copy()
    effects = [
        # 1. Geometry distortions (mask-protected)
        (params.frame_drift, apply_frame_drift, True),
        (params.melting, apply_melting, True),
        (params.block_tear, apply_block_tear, True),
        (params.datamosh, lambda f, m, i, c, idx: apply_datamosh(f, prev_frame, m, i, c, idx), True),
        (params.ghost_trail, lambda f, m, i, c, idx: apply_ghost_trail(f, prev_frame, m, i, idx), True),
        (params.flow_warp, apply_flow_warp, True),
        (params.slit_scan, lambda f, m, i, c, idx: apply_slit_scan(f, prev_frame, m, i, c, idx), True),
        (params.parallax_split, apply_parallax_split, True),
        (params.voronoi_shatter, apply_voronoi_shatter, True),
        (params.echo_crown, lambda f, m, i, c, idx: apply_echo_crown(f, prev_frame, m, i, idx), True),
        (params.strobe_phase, apply_strobe_phase, True),
        (params.frame_drop, lambda f, m, i, c, idx: apply_frame_drop(f, prev_frame, m, i, c, idx), True),
        (params.palette_cycle, apply_palette_cycle, True),
        (params.chroma_collapse, apply_chroma_collapse, True),
        (params.ordered_dither, apply_ordered_dither, True),
        (params.sigil_ring, apply_sigil_ring, True),
        (params.phylactery_glow, apply_phylactery_glow, True),
        (params.abyss_window, apply_abyss_window, True),
        (params.fft_jitter, apply_fft_jitter, True),
        (params.moire_grid, apply_moire_grid, True),
        # 2. Color distortions (mask-protected)
        (params.rgb_shift_intensity, apply_rgb_channel_shift, True),
        (params.chromatic_aberration, apply_chromatic_aberration, True),
        (params.bitcrush, apply_bitcrush, True),
        (params.chroma_dropout, apply_chroma_dropout, False),  # overlay, no displacement
        # 3. Overlays
        (params.scanlines, apply_scanlines, False),  # overlay
        (params.rolling_scanlines, apply_rolling_scanlines, False),  # overlay
        (params.neon_bars, apply_neon_bars, True),
        (params.vhs_tracking, apply_vhs_tracking, True),
        (params.crt_flicker, apply_crt_flicker, False),  # overlay
        (params.pixel_sort, apply_pixel_sort, True),  # displaces pixels
        (params.pixelation, apply_pixelation, True),
        # 4. Texture (mask-exempt overlays)
        (params.digital_noise, apply_digital_noise, False),
        (params.tv_static, apply_tv_static, False),
        (params.film_grain, apply_film_grain, False),
        # 5. Grade
        (params.vignette, lambda f, m, i, c, idx: apply_vignette(f, m, i, idx), False),
        # 6. Emissive
        (params.bloom, lambda f, m, i, c, idx: apply_bloom(f, m, i, idx), True),
    ]
    chaos = params.chaos_level
    for intensity, effect_fn, use_mask in effects:
        if intensity > 0:
            m = mask_binary if use_mask else mask_zeros
            bg_result = effect_fn(bg_result, m, intensity, chaos, frame_idx)

    if params.color_scheme and params.color_scheme != "default":
        # Color scheme is overlay-only, no displacement
        bg_result = apply_color_scheme(bg_result, params.color_scheme, mask_zeros)

    fg_result = apply_foreground_effects(frame, mask_soft, params.foreground_effect, frame_idx)
    if params.subject_invert > 0:
        fg_result = apply_subject_invert(fg_result, mask_soft, params.subject_invert, frame_idx)
    if params.subject_particles > 0:
        fg_result = apply_subject_particles(fg_result, mask_soft, params.subject_particles, frame_idx)
    # mask_opacity: 1=subject protected, 0=effects full on subject (blend both)
    opacity = np.clip(params.mask_opacity, 0.0, 1.0)
    subject_blend = (
        opacity * fg_result.astype(np.float32)
        + (1 - opacity) * bg_result.astype(np.float32)
    )
    composite = (
        mask_soft[:, :, np.newaxis] * subject_blend
        + (1 - mask_soft[:, :, np.newaxis]) * bg_result.astype(np.float32)
    ).astype(np.uint8)

    composite = apply_edge_pulse(composite, mask_soft, params.edge_pulse, frame_idx)
    if params.edge_dissolve > 0:
        composite = apply_edge_dissolve(composite, mask_binary, params.edge_dissolve, chaos, frame_idx)
    # GLSL shader chain (before final return)
    from shader_processor import shader_processor
    seed = frame_idx * 7919  # deterministic per-frame seed
    composite = shader_processor.apply_shaders_chain(composite, mask_soft, params, frame_idx, seed)
    return composite


def process_frame_boundary_aware(
    frame: np.ndarray,
    mask: np.ndarray,
    preset_name: str,
    frame_idx: int,
    prev_frame: Optional[np.ndarray],
    token_seed: int = 0,
) -> np.ndarray:
    """
    Boundary-aware pipeline: base effects (steady) + transition hit only at swap boundary.
    A/B alternation: frames 0-5 = A, 6-11 = B. Hit at frames 4, 5, 10, 11.
    """
    from glitch_preset_config import get_preset_config, get_boundary_strength, SWAP_PERIOD

    config = get_preset_config(preset_name)
    if config is None:
        return frame.copy()

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_soft = make_soft_mask(mask_binary, 2)
    glitch_region = (mask_binary < 128)

    # Base params: steady, no event spikes (for common/uncommon)
    valid_keys = set(GlitchParams.__dataclass_fields__)
    base_dict = {k: v for k, v in config.base_params.items() if k in valid_keys}
    base_params = GlitchParams(**base_dict)

    # Apply base effects (steady intensity, no per-frame random variation)
    bg_result = frame.copy().astype(np.float32)
    chaos = 0.3

    if base_params.vignette > 0:
        bg_result = apply_vignette(bg_result.astype(np.uint8), mask_binary, base_params.vignette, frame_idx).astype(np.float32)
    if base_params.film_grain > 0:
        preset_seed = abs(hash(preset_name)) % (2**32)
        tex = _get_coherent_texture(frame.shape[0], frame.shape[1], token_seed, preset_seed, frame_idx)
        amp = base_params.film_grain * 1.0
        noise = ((tex[:, :, 0] - 128) / 128 * amp).astype(np.int16)
        grained = np.clip(bg_result.astype(np.int16) + noise[:, :, np.newaxis], 0, 255).astype(np.uint8)
        bg_result = np.where(mask_binary[:, :, np.newaxis] == 255, bg_result, grained)
    if base_params.digital_noise > 0:
        bg_result = apply_digital_noise(bg_result.astype(np.uint8), mask_binary, base_params.digital_noise, chaos, frame_idx).astype(np.float32)
    if base_params.rolling_scanlines > 0:
        bg_result = apply_rolling_scanlines(bg_result.astype(np.uint8), mask_binary, base_params.rolling_scanlines, chaos, frame_idx).astype(np.float32)
    if base_params.crt_flicker > 0:
        bg_result = apply_crt_flicker(bg_result.astype(np.uint8), mask_binary, base_params.crt_flicker, chaos, frame_idx).astype(np.float32)
    if base_params.chromatic_aberration > 0:
        bg_result = apply_chromatic_aberration(bg_result.astype(np.uint8), mask_binary, base_params.chromatic_aberration, chaos, frame_idx).astype(np.float32)
    if base_params.bloom > 0:
        bg_result = apply_bloom(bg_result.astype(np.uint8), mask_binary, base_params.bloom, frame_idx).astype(np.float32)
    if base_params.vhs_tracking > 0:
        bg_result = apply_vhs_tracking(bg_result.astype(np.uint8), mask_binary, base_params.vhs_tracking, chaos, frame_idx).astype(np.float32)
    if base_params.scanlines > 0:
        bg_result = apply_scanlines(bg_result.astype(np.uint8), mask_binary, base_params.scanlines, chaos, frame_idx).astype(np.float32)
    if base_params.rgb_shift_intensity > 0:
        bg_result = apply_rgb_channel_shift(bg_result.astype(np.uint8), mask_binary, base_params.rgb_shift_intensity, chaos, frame_idx).astype(np.float32)
    if base_params.pixelation > 0:
        bg_result = apply_pixelation(bg_result.astype(np.uint8), mask_binary, base_params.pixelation, chaos, frame_idx).astype(np.float32)
    if base_params.frame_drift > 0:
        bg_result = apply_frame_drift(bg_result.astype(np.uint8), mask_binary, base_params.frame_drift, chaos, frame_idx).astype(np.float32)
    if base_params.chroma_dropout > 0:
        bg_result = apply_chroma_dropout(bg_result.astype(np.uint8), mask_binary, base_params.chroma_dropout, chaos, frame_idx).astype(np.float32)
    if base_params.edge_dissolve > 0:
        bg_result = apply_edge_dissolve(bg_result.astype(np.uint8), mask_binary, base_params.edge_dissolve, chaos, frame_idx).astype(np.float32)

    bg_result = np.clip(bg_result, 0, 255).astype(np.uint8)

    # Transition hit only at boundary
    boundary_strength = get_boundary_strength(frame_idx, config, token_seed)
    if boundary_strength > 0:
        bg_result = apply_transition_hit(
            bg_result,
            mask_binary,
            boundary_strength,
            frame_idx,
            token_seed,
            preset_name,
            prev_frame=prev_frame,
            hit_components=config.hit_components,
        )

    # Composite: subject from original (with optional subject effects), background from processed
    subject = frame.copy()
    if base_params.subject_particles > 0:
        subject = apply_subject_particles(subject, mask_soft, base_params.subject_particles, frame_idx)
    composite = (
        mask_soft[:, :, np.newaxis] * subject.astype(np.float32)
        + (1 - mask_soft[:, :, np.newaxis]) * bg_result.astype(np.float32)
    ).astype(np.uint8)
    if base_params.edge_pulse > 0:
        composite = apply_edge_pulse(composite, mask_soft, base_params.edge_pulse, frame_idx)
    # GLSL shader chain
    from shader_processor import shader_processor
    composite = shader_processor.apply_shaders_chain(composite, mask_soft, base_params, frame_idx, token_seed)
    return composite
