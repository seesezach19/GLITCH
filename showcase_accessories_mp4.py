"""
Generate an MP4 showcasing all accessories.
Uses BIOLUME_VOID_UNCOMMON background with LICHE_BASE and LICHE_RED_BLACK.
chromatic_bloom_uncommon preset. Each accessory gets 2.5 seconds.
Usage: python showcase_accessories_mp4.py [-o output/showcase_accessories.mp4]
"""
import argparse
import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

from glitch_processor import GlitchParams
from gif_processor import process_images_to_gif, GIF_TARGET_BYTES
from series_generator import (
    SERIES_ROOT,
    _collect_layer_paths,
    _composite_layers,
    ACCESSORY_LAYER_ORDER,
    BEHIND_BASE_ACCESSORIES,
)

SAVED_PRESETS_PATH = Path(__file__).resolve().parent / "saved_presets.json"
FRAME_DURATION_MS = 300
STATIC_FRAMES_PER_IMAGE = 6


def _params_from_preset(name: str) -> GlitchParams:
    """Build GlitchParams from saved preset."""
    if not SAVED_PRESETS_PATH.exists():
        return GlitchParams()
    with open(SAVED_PRESETS_PATH, encoding="utf-8") as f:
        saved = json.load(f)
    p = saved.get(name, {})
    valid = set(GlitchParams.__dataclass_fields__)
    filtered = {k: v for k, v in p.items() if k in valid}
    return GlitchParams(**filtered)


def _find_layer_path(layers: dict, name: str, category: str) -> Path | None:
    """Find layer by stem name."""
    for p in layers.get(category, []):
        if p.stem.upper() == name.upper():
            return p
    return None


def _get_all_accessories(layers: dict) -> list[tuple[str, Path, bool]]:
    """Return list of (display_name, path, behind_base) for each accessory, in layer order."""
    result = []
    for key in ACCESSORY_LAYER_ORDER:
        if key not in layers or not layers[key]:
            continue
        for p in layers[key]:
            behind = p.stem.lower() in BEHIND_BASE_ACCESSORIES
            result.append((p.stem, p, behind))
    # Any extra accessory categories
    for key in sorted(layers.keys()):
        if key in ("base", "alt_bases", "backgrounds", "eyes") or key in ACCESSORY_LAYER_ORDER:
            continue
        if not layers[key]:
            continue
        for p in layers[key]:
            behind = p.stem.lower() in BEHIND_BASE_ACCESSORIES
            result.append((p.stem, p, behind))
    return result


def main():
    parser = argparse.ArgumentParser(description="Showcase MP4: all accessories")
    parser.add_argument("-o", "--output", default="output/showcase_accessories.mp4", help="Output MP4 path")
    parser.add_argument("--duration", type=float, default=2.5, help="Seconds per accessory")
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    layers = _collect_layer_paths()

    bg_path = SERIES_ROOT / "BACKGROUNDS" / "BIOLUME_VOID_UNCOMMON.png"
    if not bg_path.exists():
        print(f"Error: {bg_path} not found")
        return 1

    base_path = _find_layer_path(layers, "LICHE_BASE", "base") or (SERIES_ROOT / "LICHE_BASE.png")
    alt_path = _find_layer_path(layers, "LICHE_RED_BLACK", "alt_bases")

    if not base_path.exists():
        print(f"Error: LICHE_BASE not found at {base_path}")
        return 1
    if not alt_path or not alt_path.exists():
        print(f"Error: LICHE_RED_BLACK not found")
        return 1

    accessories = _get_all_accessories(layers)
    if not accessories:
        print("Error: No accessories found in SERIES/ACCESSORIES")
        return 1

    with open(SAVED_PRESETS_PATH, encoding="utf-8") as f:
        saved_presets = json.load(f)
    bypass_mask = saved_presets.get("chromatic_bloom_uncommon", {}).get("bypass_mask", True)
    params = _params_from_preset("chromatic_bloom_uncommon")

    mask_path = None
    for name in ("LICHE_MASK.png", "mask.png", "MASK.png"):
        p = SERIES_ROOT / name
        if p.exists():
            mask_path = str(p)
            break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    background = PILImage.open(bg_path).convert("RGBA")
    w, h = background.size

    base_img = PILImage.open(base_path).convert("RGBA")
    if base_img.size != (w, h):
        base_img = base_img.resize((w, h), PILImage.Resampling.LANCZOS)
    alt_img = PILImage.open(alt_path).convert("RGBA")
    if alt_img.size != (w, h):
        alt_img = alt_img.resize((w, h), PILImage.Resampling.LANCZOS)

    eyes = PILImage.new("RGBA", (w, h), (0, 0, 0, 0))

    temp_dir = Path(tempfile.mkdtemp(prefix="showcase_accessories_"))
    segment_gifs: list[Path] = []

    try:
        for i, (acc_name, acc_path, behind) in enumerate(accessories):
            acc_img = PILImage.open(acc_path).convert("RGBA")
            if acc_img.size != (w, h):
                acc_img = acc_img.resize((w, h), PILImage.Resampling.LANCZOS)

            if behind:
                acc_behind = [acc_img]
                acc_front = []
            else:
                acc_behind = []
                acc_front = [acc_img]

            img_base = _composite_layers(background, base_img, eyes, acc_front, acc_behind)
            img_alt = _composite_layers(background, alt_img, eyes, acc_front, acc_behind)
            img_base = img_base.resize((512, 512), PILImage.Resampling.LANCZOS)
            img_alt = img_alt.resize((512, 512), PILImage.Resampling.LANCZOS)

            base_f = str(temp_dir / "base.png")
            alt_f = str(temp_dir / "alt.png")
            cv2.imwrite(base_f, cv2.cvtColor(np.array(img_base), cv2.COLOR_RGB2BGR))
            cv2.imwrite(alt_f, cv2.cvtColor(np.array(img_alt), cv2.COLOR_RGB2BGR))

            gif_path = temp_dir / f"seg_{i:03d}.gif"
            print(f"[{i + 1}/{len(accessories)}] {acc_name}...", end=" ", flush=True)

            process_images_to_gif(
                [base_f, alt_f],
                str(gif_path),
                params,
                frame_duration_ms=FRAME_DURATION_MS,
                static_frames_per_image=STATIC_FRAMES_PER_IMAGE,
                bypass_mask=bypass_mask,
                mask_path=mask_path,
                max_bytes=GIF_TARGET_BYTES,
            )
            segment_gifs.append(gif_path)
            print("done")

        from moviepy import VideoFileClip, concatenate_videoclips

        clips = []
        target_duration = args.duration

        for gif_path in segment_gifs:
            clip = VideoFileClip(str(gif_path))
            if clip.duration < target_duration:
                n_loops = math.ceil(target_duration / clip.duration)
                clip = concatenate_videoclips([clip] * n_loops)
            clip = clip.subclip(0, target_duration) if hasattr(clip, "subclip") else clip.subclipped(0, target_duration)
            clips.append(clip)
            clip.close()

        print("Stitching segments to MP4...")
        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(
            str(out_path),
            codec="libx264",
            fps=args.fps,
            preset="medium",
            ffmpeg_params=["-pix_fmt", "yuv420p"],
            logger=None,
        )
        final.close()

        print(f"Done: {out_path} ({len(clips)} accessories, {args.duration}s each)")

    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    exit(main())
