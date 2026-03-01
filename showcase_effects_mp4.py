"""
Generate an MP4 showcasing glitch presets in order: common → uncommon → rare → legendary.
Uses BLOODMIST_VOID_LEGENDARY background with LICHE_BASE and LICHE_CYAN (alt).
Each preset is shown for 2.5 seconds.
Usage: python showcase_effects_mp4.py [-o output/showcase_effects.mp4]
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
from series_generator import SERIES_ROOT, _collect_layer_paths, _composite_layers

RARITY_ORDER = ("common", "uncommon", "rare", "super_rare", "legendary")
SAVED_PRESETS_PATH = Path(__file__).resolve().parent / "saved_presets.json"
FRAME_DURATION_MS = 300
STATIC_FRAMES_PER_IMAGE = 6


def _get_presets_by_rarity() -> list[str]:
    """Return preset names from saved_presets.json, sorted common → legendary."""
    if not SAVED_PRESETS_PATH.exists():
        return []
    with open(SAVED_PRESETS_PATH, encoding="utf-8") as f:
        presets = json.load(f)
    # Sort by rarity suffix
    def rarity_rank(name: str) -> int:
        name_lower = name.lower()
        for i, r in enumerate(RARITY_ORDER):
            if name_lower.endswith("_" + r):
                return i
        return 999

    return sorted(presets.keys(), key=lambda n: (rarity_rank(n), n))


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
    """Find layer by stem name (e.g. LICHE_CYAN)."""
    for p in layers.get(category, []):
        if p.stem.upper() == name.upper():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Showcase MP4: glitch presets (common → legendary)")
    parser.add_argument("-o", "--output", default="output/showcase_effects.mp4", help="Output MP4 path")
    parser.add_argument("--duration", type=float, default=2.5, help="Seconds per preset")
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    preset_names = _get_presets_by_rarity()
    if not preset_names:
        print("Error: No presets found in saved_presets.json")
        return 1

    layers = _collect_layer_paths()

    bg_path = SERIES_ROOT / "BACKGROUNDS" / "BLOODMIST_VOID_LEGENDARY.png"
    if not bg_path.exists():
        print(f"Error: {bg_path} not found")
        return 1

    base_path = _find_layer_path(layers, "LICHE_BASE", "base") or (SERIES_ROOT / "LICHE_BASE.png")
    alt_path = _find_layer_path(layers, "LICHE_CYAN_WHITE", "alt_bases")
    if not alt_path:
        alt_path = _find_layer_path(layers, "LICHE_CYAN", "alt_bases")
    if not alt_path and layers.get("alt_bases"):
        alt_path = layers["alt_bases"][0]

    if not base_path.exists():
        print(f"Error: LICHE_BASE not found at {base_path}")
        return 1
    if not alt_path or not alt_path.exists():
        print(f"Error: LICHE_CYAN / LICHE_CYAN_WHITE not found")
        return 1

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
    img_base = _composite_layers(background, base_img, eyes, [])
    img_alt = _composite_layers(background, alt_img, eyes, [])
    img_base = img_base.resize((512, 512), PILImage.Resampling.LANCZOS)
    img_alt = img_alt.resize((512, 512), PILImage.Resampling.LANCZOS)

    temp_dir = Path(tempfile.mkdtemp(prefix="showcase_effects_"))
    base_f = str(temp_dir / "base.png")
    alt_f = str(temp_dir / "alt.png")
    cv2.imwrite(base_f, cv2.cvtColor(np.array(img_base), cv2.COLOR_RGB2BGR))
    cv2.imwrite(alt_f, cv2.cvtColor(np.array(img_alt), cv2.COLOR_RGB2BGR))

    bypass_mask = True
    segment_gifs: list[Path] = []

    with open(SAVED_PRESETS_PATH, encoding="utf-8") as f:
        saved_presets = json.load(f)

    try:
        for i, preset_name in enumerate(preset_names):
            params = _params_from_preset(preset_name)
            bypass_mask = saved_presets.get(preset_name, {}).get("bypass_mask", True)

            gif_path = temp_dir / f"seg_{i:03d}.gif"
            print(f"[{i + 1}/{len(preset_names)}] {preset_name}...", end=" ", flush=True)

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

        print(f"Done: {out_path} ({len(clips)} presets, {args.duration}s each)")

    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    exit(main())
