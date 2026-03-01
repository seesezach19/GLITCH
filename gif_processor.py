"""
Liche - GIF Processing
Images → glitched animated GIF. Keeps output under 15MB.
"""

from __future__ import annotations

import os
import random
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image as PILImage

from glitch_processor import (
    GlitchParams,
    create_auto_mask,
    process_frame,
    process_frame_boundary_aware,
    vary_params_for_frame,
)
from transition_effects import apply_transition, pick_transition

GIF_MAX_BYTES = 15 * 1024 * 1024  # 15 MB limit
GIF_TARGET_BYTES = 700 * 1024  # 700 KB target for Series Generator
GIF_TARGET_SIZE = (512, 512)  # Default output size

RESOLUTION_OPTIONS = {
    "256x256": (256, 256),
    "512x512": (512, 512),
    "768x768": (768, 768),
    "1024x1024": (1024, 1024),
}

# Background effects (apply to full image when mask bypassed)
BACKGROUND_EFFECT_KEYS = [
    "rgb_shift_intensity",
    "chromatic_aberration",
    "scanlines",
    "digital_noise",
    "pixelation",
    "datamosh",
    "melting",
    "vhs_tracking",
    "crt_flicker",
    "bitcrush",
    "tv_static",
    "frame_drift",
    "rolling_scanlines",
    "ghost_trail",
    "block_tear",
    "vignette",
    "bloom",
    "film_grain",
    "pixel_sort",
    "neon_bars",
    "edge_dissolve",
    "chroma_dropout",
    "flow_warp",
    "slit_scan",
    "parallax_split",
    "voronoi_shatter",
    "echo_crown",
    "strobe_phase",
    "frame_drop",
    "palette_cycle",
    "chroma_collapse",
    "ordered_dither",
    "sigil_ring",
    "phylactery_glow",
    "abyss_window",
    "fft_jitter",
    "moire_grid",
    "shader_necrotic_iridescent_flow_intensity",
    "shader_hexagonal_warp_intensity",
    "shader_caustic_flow_intensity",
    "shader_thermal_distort_intensity",
    "shader_pixel_rain_intensity",
    "shader_liquid_metal_intensity",
    "shader_data_corruption_intensity",
    "shader_vhs_rewind_intensity",
    "shader_holographic_foil_intensity",
    "displacement_map",
    "color_halftone",
    "temporal_echo",
]

# Foreground-only effects (require mask; ignored when bypass_mask=True)
FOREGROUND_EFFECT_KEYS = [
    "foreground_effect",
    "subject_invert",
    "subject_particles",
    "edge_pulse",
]


def random_glitch_params(
    num_effects: int = 5,
    seed: Optional[int] = None,
    background_only: bool = False,
) -> GlitchParams:
    """Return GlitchParams with num_effects random effects at random levels (1-3)."""
    if seed is not None:
        random.seed(seed)
    pool = BACKGROUND_EFFECT_KEYS if background_only else BACKGROUND_EFFECT_KEYS + FOREGROUND_EFFECT_KEYS
    base = {k: 0 for k in pool}
    chosen = random.sample(pool, min(num_effects, len(pool)))
    for k in chosen:
        base[k] = random.randint(1, 3)
    base["chaos_level"] = random.uniform(0.3, 0.9)
    base["mask_sensitivity"] = random.uniform(0.0, 0.3)
    base["color_scheme"] = random.choice(["default", "cold", "warm", "neon", "vhs", "blood"])
    return GlitchParams(**base)


def _effects_disabled(params: GlitchParams) -> bool:
    """True when all effects are off — safe to pass through unchanged."""
    return (
        params.rgb_shift_intensity == 0
        and params.chromatic_aberration == 0
        and params.scanlines == 0
        and params.digital_noise == 0
        and params.pixelation == 0
        and params.datamosh == 0
        and params.melting == 0
        and params.vhs_tracking == 0
        and params.crt_flicker == 0
        and params.bitcrush == 0
        and params.tv_static == 0
        and params.frame_drift == 0
        and params.rolling_scanlines == 0
        and params.ghost_trail == 0
        and params.block_tear == 0
        and params.vignette == 0
        and params.bloom == 0
        and params.film_grain == 0
        and params.pixel_sort == 0
        and params.neon_bars == 0
        and params.edge_dissolve == 0
        and params.chroma_dropout == 0
        and params.foreground_effect == 0
        and params.subject_invert == 0
        and params.subject_particles == 0
        and params.edge_pulse == 0
        and params.flow_warp == 0
        and params.slit_scan == 0
        and params.parallax_split == 0
        and params.voronoi_shatter == 0
        and params.echo_crown == 0
        and params.strobe_phase == 0
        and params.frame_drop == 0
        and params.palette_cycle == 0
        and params.chroma_collapse == 0
        and params.ordered_dither == 0
        and params.sigil_ring == 0
        and params.phylactery_glow == 0
        and params.abyss_window == 0
        and params.fft_jitter == 0
        and params.moire_grid == 0
        and params.shader_necrotic_iridescent_flow_intensity == 0
        and params.shader_hexagonal_warp_intensity == 0
        and params.shader_caustic_flow_intensity == 0
        and params.shader_thermal_distort_intensity == 0
        and params.shader_void_tendrils_intensity == 0
        and params.shader_spectral_prism_intensity == 0
        and params.shader_soul_fire_intensity == 0
        and params.shader_electric_arc_intensity == 0
        and params.shader_dimensional_rift_intensity == 0
        and params.shader_glitch_hologram_intensity == 0
        and params.shader_crystalline_frost_intensity == 0
        and params.shader_pixel_rain_intensity == 0
        and params.shader_liquid_metal_intensity == 0
        and params.shader_data_corruption_intensity == 0
        and params.shader_vhs_rewind_intensity == 0
        and params.shader_holographic_foil_intensity == 0
        and params.shader_kaleido_grid_intensity == 0
        and params.shader_stripe_shift_intensity == 0
        and params.shader_block_smear_intensity == 0
        and params.shader_palette_rainbow_intensity == 0
        and params.displacement_map == 0
        and params.color_halftone == 0
        and params.temporal_echo == 0
        and params.color_scheme in (None, "", "default", "custom")
    )


def _build_global_palette(frames: list, colors: int = 256) -> "PILImage.Image | None":
    """Build one palette from all frames using random sampling for better color coverage."""
    if not frames or colors < 2:
        return None
    h, w = frames[0].size[1], frames[0].size[0]
    n_samples_per_frame = min(h * w, max(colors * 8, 4096))
    rng = np.random.RandomState(42)
    all_pixels = []
    for img in frames:
        rgb = img.convert("RGB") if img.mode != "RGB" else img
        arr = np.array(rgb).reshape(-1, 3)
        indices = rng.choice(len(arr), min(n_samples_per_frame, len(arr)), replace=False)
        all_pixels.append(arr[indices])
    pixels = np.concatenate(all_pixels, axis=0)
    pixels = pixels[:min(len(pixels), 512 * 512)]
    if len(pixels) < colors:
        return None
    n = len(pixels)
    side = int(n ** 0.5)
    combined = PILImage.fromarray(pixels[:side * side].reshape(side, side, 3), mode="RGB")
    return combined.quantize(colors=colors, method=2)


def _frame_similarity(frames: list) -> list[float]:
    """Compute mean absolute difference between consecutive frames. Lower = more similar."""
    diffs = []
    for i in range(1, len(frames)):
        a = np.array(frames[i - 1].convert("RGB"), dtype=np.float32)
        b = np.array(frames[i].convert("RGB"), dtype=np.float32)
        diffs.append(float(np.mean(np.abs(a - b))))
    return diffs


def _drop_similar_frames(frames: list, target_count: int) -> tuple[list, int]:
    """Drop most-similar consecutive frames, return (reduced_frames, duration_multiplier)."""
    if len(frames) <= target_count:
        return frames, 1
    diffs = _frame_similarity(frames)
    n_drop = len(frames) - target_count
    drop_indices = set()
    for _ in range(n_drop):
        best_i = None
        best_val = float("inf")
        for i, d in enumerate(diffs):
            if (i + 1) not in drop_indices and d < best_val:
                best_val = d
                best_i = i + 1
        if best_i is not None:
            drop_indices.add(best_i)
    result = [f for i, f in enumerate(frames) if i not in drop_indices]
    multiplier = max(1, round(len(frames) / max(1, len(result))))
    return result, multiplier


def _save_gif_under_size(
    pil_frames: list,
    output_path: str,
    duration: int,
    max_bytes: int = GIF_MAX_BYTES,
    target_size: tuple[int, int] | None = GIF_TARGET_SIZE,
    global_palette: bool = True,
    use_dithering: bool = False,
) -> str:
    """Save GIF with global palette. Smarter frame decimation drops least-changed frames first."""
    dither_mode = 1 if use_dithering else 0

    def save_frames(frames: list, path: str, dur: int, colors: int = 256, palette_ref: "PILImage.Image | None" = None) -> int:
        if not frames:
            return 0
        if palette_ref is not None and palette_ref.mode == "P":
            quantized = []
            try:
                for f in frames:
                    rgb = f.convert("RGB") if f.mode != "RGB" else f
                    q = rgb.quantize(palette=palette_ref, dither=dither_mode)
                    quantized.append(q)
            except (ValueError, TypeError, OSError):
                palette_ref = None
        if palette_ref is None:
            quantized = [f.quantize(colors=colors, method=2, dither=dither_mode)
                         if f.mode != "P" else f for f in frames]
        quantized[0].save(
            path,
            save_all=True,
            append_images=quantized[1:],
            duration=dur,
            loop=0,
            optimize=True,
        )
        return os.path.getsize(path)

    frames = [f.copy() for f in pil_frames]
    resample = PILImage.LANCZOS

    if target_size and frames[0].size != target_size:
        frames = [f.resize(target_size, resample) for f in frames]

    def try_save(frames_list: list, dur: int, n_colors: int) -> int:
        palette_ref = None
        if global_palette:
            try:
                palette_ref = _build_global_palette(frames_list, n_colors)
            except Exception:
                palette_ref = None
        return save_frames(frames_list, output_path, dur, n_colors, palette_ref)

    for colors in (256, 128, 64, 32):
        size = try_save(frames, duration, colors)
        if size <= max_bytes:
            return output_path

    reduced, mult = _drop_similar_frames(frames, len(frames) * 2 // 3)
    for colors in (128, 64, 32):
        size = try_save(reduced, duration * mult, colors)
        if size <= max_bytes:
            return output_path

    reduced_more, mult2 = _drop_similar_frames(frames, len(frames) // 3)
    try_save(reduced_more, duration * mult2, 32)
    return output_path


def _load_images(paths: list[str]) -> list[np.ndarray]:
    """Load images as BGR, resized to match first image."""
    frames: list[np.ndarray] = []
    ref_h, ref_w = None, None
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            with open(path, "rb") as f:
                img = cv2.imdecode(
                    np.frombuffer(f.read(), dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
        if img is None:
            continue
        if ref_h is None:
            ref_h, ref_w = img.shape[:2]
        if img.shape[:2] != (ref_h, ref_w):
            img = cv2.resize(img, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
        frames.append(img)
    return frames


def _load_mask_for_frame(mask_path: str, frame_shape: tuple[int, int]) -> np.ndarray:
    """Load mask from path, resize to frame size. Area inside the drawn region = 255 (protected), outside = 0 (glitch)."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    h, w = frame_shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    # Any non-black pixel = protected (handles black bg + colored subject, e.g. magenta)
    _, mask_bin = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    return mask_bin


def process_images_to_gif(
    image_paths: list[str],
    output_path: str,
    params: GlitchParams,
    frame_duration_ms: int = 200,
    static_frames_per_image: int = 12,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    bypass_mask: bool = False,
    mask_path: Optional[str] = None,
    max_bytes: Optional[int] = None,
    output_size: Optional[tuple[int, int]] = None,
    use_dithering: bool = False,
) -> str:
    """
    Build glitched GIF from images. Zero effects → clean GIF from originals.
    bypass_mask: if True, apply effects to full image (no foreground protection).
    mask_path: if set, use this mask file (255=protected, 0=glitch) instead of auto-mask.
    output_size: (w, h) for output resolution, defaults to GIF_TARGET_SIZE.
    use_dithering: if True, apply Floyd-Steinberg dithering for smoother color gradients.
    """
    target_sz = output_size or GIF_TARGET_SIZE
    if not image_paths:
        raise ValueError("No images provided")

    if _effects_disabled(params):
        frames = _load_images(image_paths)
        if not frames:
            raise ValueError("Could not load any images")
        pil = [PILImage.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        sub_dur = max(20, frame_duration_ms // static_frames_per_image)
        expanded = [p for p in pil for _ in range(static_frames_per_image)]
        _save_gif_under_size(expanded, output_path, sub_dur, max_bytes or GIF_MAX_BYTES,
                             target_size=target_sz, global_palette=True, use_dithering=use_dithering)
        if progress_callback:
            progress_callback(1.0, "Zero effects — copied to GIF")
        return output_path

    frames_bgr = _load_images(image_paths)
    if not frames_bgr:
        raise ValueError("Could not load any images")

    from shader_processor import shader_processor
    shader_processor.reset_feedback()

    n_images = len(frames_bgr)
    total = n_images * static_frames_per_image
    sub_dur = max(20, frame_duration_ms // static_frames_per_image)
    transition_frames = min(3, static_frames_per_image // 2)

    processed: list[np.ndarray] = []
    prev: Optional[np.ndarray] = None

    for i, frame in enumerate(frames_bgr):
        if bypass_mask:
            mask_bin = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        elif mask_path:
            mask_bin = _load_mask_for_frame(mask_path, frame.shape)
        else:
            mask_bin = create_auto_mask(frame, use_background_subtraction=True, sensitivity=0.5, face_safe=True)
            _, mask_bin = cv2.threshold(mask_bin, 127, 255, cv2.THRESH_BINARY)

        for k in range(static_frames_per_image):
            idx = i * static_frames_per_image + k
            fp = vary_params_for_frame(params, idx)
            is_transition = i > 0 and k < transition_frames and prev is not None
            frame_to_process = frame
            if is_transition:
                t = (k + 1) / (transition_frames + 1)
                weights = {
                    "band_wipe": fp.transition_band_wipe,
                    "diagonal_rip": fp.transition_diagonal_rip,
                    "slit_scan_swap": fp.transition_slit_scan_swap,
                    "pixel_scramble_patch": fp.transition_pixel_scramble_patch,
                    "edge_first_reveal": fp.transition_edge_first_reveal,
                    "voronoi_shatter_swap": fp.transition_voronoi_shatter_swap,
                    "phase_offset_echo": fp.transition_phase_offset_echo,
                    "palette_snap_posterize": fp.transition_palette_snap_posterize,
                    "chroma_dropout_rebound": fp.transition_chroma_dropout_rebound,
                    "scanline_gate": fp.transition_scanline_gate,
                    "micro_jitter_rgb": fp.transition_micro_jitter_rgb,
                    "noise_threshold_crossfade": fp.transition_noise_threshold_crossfade,
                }
                chosen = pick_transition(weights, seed=idx * 7907)
                if chosen:
                    intensity = weights.get(chosen, 1)
                    frame_to_process = apply_transition(
                        prev, frame, mask_bin, t=t,
                        transition_name=chosen, intensity=min(10, intensity), seed=idx * 3313,
                    )
            out = process_frame(frame_to_process, mask_bin, fp, idx, prev)
            processed.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            if progress_callback:
                progress_callback((idx + 1) / total, f"Image {i + 1}/{n_images} frame {k + 1}/{static_frames_per_image}")
        prev = frame

    pil_frames = [PILImage.fromarray(p) for p in processed]
    _save_gif_under_size(pil_frames, output_path, sub_dur, max_bytes or GIF_MAX_BYTES,
                         target_size=target_sz, global_palette=True, use_dithering=use_dithering)
    return output_path


def process_images_to_gif_boundary_aware(
    image_paths: list[str],
    output_path: str,
    preset_name: str,
    token_seed: int = 0,
    frame_duration_ms: int = 50,
    static_frames_per_image: int = 6,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    mask_path: Optional[str] = None,
    max_bytes: Optional[int] = None,
    save_debug_frames: bool = False,
) -> str:
    """
    Boundary-aware GIF: A/B alternation (6 frames each), transition hit only at swap boundary.
    image_paths: [A_path, B_path] - A=black/white, B=cyan/black skeleton.
    Always outputs 12 frames, 50ms each.
    bypass_mask from preset: if True, effects apply to full image (no mask). mask_path optional when bypassing.
    """
    from glitch_preset_config import SWAP_PERIOD, TOTAL_FRAMES, FRAME_DURATION_MS, get_preset_config

    if not image_paths or len(image_paths) < 2:
        raise ValueError("Need exactly 2 images (A and B base)")

    config = get_preset_config(preset_name)
    bypass_mask = config.bypass_mask if config else True
    if not bypass_mask and mask_path is None:
        raise ValueError("mask_path required when preset has bypass_mask=False")

    frames_bgr = _load_images(image_paths[:2])
    if not frames_bgr or len(frames_bgr) < 2:
        raise ValueError("Could not load A and B images")

    from shader_processor import shader_processor
    shader_processor.reset_feedback()

    frame_a, frame_b = frames_bgr[0], frames_bgr[1]
    if bypass_mask:
        mask_bin = np.zeros((frame_a.shape[0], frame_a.shape[1]), dtype=np.uint8)
    else:
        mask_bin = _load_mask_for_frame(mask_path, frame_a.shape)

    total = TOTAL_FRAMES
    sub_dur = FRAME_DURATION_MS

    processed: list[np.ndarray] = []
    out_dir = None
    if save_debug_frames:
        from pathlib import Path
        out_dir = Path(output_path).parent / "debug_frames"
        out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(TOTAL_FRAMES):
        phase = (idx // SWAP_PERIOD) % 2
        base_frame = frame_b if phase == 1 else frame_a
        if idx == 0:
            prev_frame = frame_b
        elif idx <= SWAP_PERIOD:
            prev_frame = frame_a
        else:
            prev_frame = frame_b

        out = process_frame_boundary_aware(
            base_frame,
            mask_bin,
            preset_name,
            idx,
            prev_frame,
            token_seed=token_seed,
        )
        processed.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        if progress_callback:
            progress_callback((idx + 1) / total, f"Frame {idx + 1}/{total}")
        if save_debug_frames and out_dir:
            pil_out = PILImage.fromarray(processed[-1])
            pil_out.save(out_dir / f"frame_{idx:02d}.png")

    pil_frames = [PILImage.fromarray(p) for p in processed]
    _save_gif_under_size(pil_frames, output_path, sub_dur, max_bytes or GIF_MAX_BYTES, global_palette=True)
    return output_path


def save_as_webp(
    pil_frames: list,
    output_path: str,
    duration: int,
    quality: int = 80,
    target_size: tuple[int, int] | None = GIF_TARGET_SIZE,
) -> str:
    """Save animation as WebP (24-bit color, much smaller than GIF)."""
    frames = [f.copy() for f in pil_frames]
    if target_size and frames[0].size != target_size:
        frames = [f.resize(target_size, PILImage.LANCZOS) for f in frames]
    rgb_frames = [f.convert("RGB") if f.mode != "RGB" else f for f in frames]
    rgb_frames[0].save(
        output_path,
        save_all=True,
        append_images=rgb_frames[1:],
        duration=duration,
        loop=0,
        quality=quality,
    )
    return output_path


def save_as_apng(
    pil_frames: list,
    output_path: str,
    duration: int,
    target_size: tuple[int, int] | None = GIF_TARGET_SIZE,
) -> str:
    """Save animation as APNG (full 24-bit color, no palette limitation)."""
    frames = [f.copy() for f in pil_frames]
    if target_size and frames[0].size != target_size:
        frames = [f.resize(target_size, PILImage.LANCZOS) for f in frames]
    rgba_frames = [f.convert("RGBA") if f.mode != "RGBA" else f for f in frames]
    rgba_frames[0].save(
        output_path,
        save_all=True,
        append_images=rgba_frames[1:],
        duration=duration,
        loop=0,
    )
    return output_path


def convert_gif_to_mp4(gif_path: str, mp4_path: str, fps: int = 15) -> str:
    """Convert animated GIF to MP4."""
    from moviepy import VideoFileClip

    clip = VideoFileClip(gif_path)
    clip.write_videofile(
        mp4_path,
        codec="libx264",
        fps=fps,
        preset="medium",
        ffmpeg_params=["-pix_fmt", "yuv420p"],
        logger=None,
    )
    clip.close()
    return mp4_path
