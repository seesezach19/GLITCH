"""
GIF Overlay - Composite a GIF onto a base image using a mask.
Scale-to-cover: GIF is scaled to fill the mask region (preserve AR, may extend beyond boundary).
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

from glitch_processor import GlitchParams, process_frame


def _load_preset(name: str, presets_path: Path) -> GlitchParams | None:
    """Load preset from saved_presets.json. Returns None if not found."""
    if not presets_path.exists():
        return None
    try:
        with open(presets_path, encoding="utf-8") as f:
            saved = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    p = saved.get(name, {})
    valid = set(GlitchParams.__dataclass_fields__)
    filtered = {k: v for k, v in p.items() if k in valid}
    if not filtered:
        return None
    return GlitchParams(**filtered)


def overlay_gif_on_image(
    base_path: str,
    gif_path: str,
    mask_path: str,
    output_path: str,
    progress_callback=None,
    base_preset_name: str | None = None,
    presets_path: Path | None = None,
) -> str:
    """
    Composite GIF onto base image using mask.
    White in mask = where GIF shows. Scale-to-cover: GIF fills mask region, preserves AR,
    may extend slightly beyond boundary (mask clips overflow).
    If base_preset_name is set, apply that preset to the base image (static parts) first.
    """
    base = cv2.imread(str(base_path))
    if base is None:
        with open(base_path, "rb") as f:
            base = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if base is None:
        raise ValueError(f"Could not load base image: {base_path}")

    # Optionally apply preset to base image (chromatic bloom, etc.)
    if base_preset_name and presets_path:
        params = _load_preset(base_preset_name, presets_path)
        if params is not None:
            # Override ordered_dither to 1 for overlay (softer dither on static image)
            params = GlitchParams(**{**vars(params), "ordered_dither": 1})
            mask_zeros = np.zeros((base.shape[0], base.shape[1]), dtype=np.uint8)
            base = process_frame(base, mask_zeros, params, frame_idx=0, prev_frame=None)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        with open(mask_path, "rb") as f:
            mask = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")

    h_base, w_base = base.shape[:2]
    if mask.shape[:2] != (h_base, w_base):
        mask = cv2.resize(mask, (w_base, h_base), interpolation=cv2.INTER_NEAREST)

    # Bounding box of mask region (white = GIF area)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Mask has no white pixels (no GIF region defined)")

    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    bbox_cx = (x_min + x_max) // 2
    bbox_cy = (y_min + y_max) // 2

    # Load GIF frames
    gif = PILImage.open(gif_path)
    frames = []
    durations = []
    try:
        while True:
            frame = gif.convert("RGBA")
            frames.append(np.array(frame))
            dur = gif.info.get("duration", 50)
            durations.append(max(20, dur))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass

    if not frames:
        raise ValueError(f"No frames in GIF: {gif_path}")

    gif_h, gif_w = frames[0].shape[:2]
    # Scale to cover: fill mask region, preserve AR, may overflow
    scale = max(bbox_w / gif_w, bbox_h / gif_h)
    new_w = int(round(gif_w * scale))
    new_h = int(round(gif_h * scale))

    # Center the scaled GIF over the mask bbox
    paste_x = bbox_cx - new_w // 2
    paste_y = bbox_cy - new_h // 2

    mask_norm = mask_bin.astype(np.float32) / 255.0
    mask_3ch = mask_norm[:, :, np.newaxis]

    out_frames = []
    for i, frame in enumerate(frames):
        if progress_callback:
            progress_callback((i + 1) / len(frames), f"Frame {i + 1}/{len(frames)}")

        # Resize frame (RGBA)
        if frame.shape[:2] != (new_h, new_w):
            frame_pil = PILImage.fromarray(frame)
            frame_pil = frame_pil.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
            frame = np.array(frame_pil)

        # Create full-size overlay: base with GIF pasted
        overlay_bgr = base.copy().astype(np.float32)
        gif_rgb = frame[:, :, :3]
        gif_alpha = frame[:, :, 3:4].astype(np.float32) / 255.0 if frame.shape[2] == 4 else np.ones((new_h, new_w, 1), dtype=np.float32)
        gif_bgr = cv2.cvtColor(gif_rgb, cv2.COLOR_RGB2BGR)

        # Paste region: clip to image bounds
        src_y0 = max(0, -paste_y)
        src_x0 = max(0, -paste_x)
        src_y1 = min(new_h, h_base - paste_y)
        src_x1 = min(new_w, w_base - paste_x)

        dst_y0 = max(0, paste_y)
        dst_x0 = max(0, paste_x)
        dst_y1 = min(h_base, paste_y + new_h)
        dst_x1 = min(w_base, paste_x + new_w)

        if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
            out_frames.append(base.copy())
            continue

        # Extract the mask region that overlaps our paste
        region_mask = mask_norm[dst_y0:dst_y1, dst_x0:dst_x1]
        region_mask_3ch = region_mask[:, :, np.newaxis]

        # Blend: use GIF where mask is white, base elsewhere
        gif_patch = gif_bgr[src_y0:src_y1, src_x0:src_x1].astype(np.float32)
        base_patch = overlay_bgr[dst_y0:dst_y1, dst_x0:dst_x1]
        alpha_patch = gif_alpha[src_y0:src_y1, src_x0:src_x1]

        # Combined alpha: mask * gif_alpha (both must be 1 to show GIF)
        combined_alpha = region_mask_3ch * alpha_patch
        blended = base_patch * (1 - combined_alpha) + gif_patch * combined_alpha
        overlay_bgr[dst_y0:dst_y1, dst_x0:dst_x1] = blended

        out_frames.append(np.clip(overlay_bgr, 0, 255).astype(np.uint8))

    # Save as GIF
    pil_frames = [PILImage.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in out_frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return output_path
