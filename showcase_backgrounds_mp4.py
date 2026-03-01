"""
Generate an MP4 showcase of all backgrounds with gentle_pulse_common preset.
Each background is shown for 2.5 seconds.
Uses process_images_to_gif with GlitchParams from saved_presets.json (same as batch_gif).
Usage: python showcase_backgrounds_mp4.py [-o output/showcase.mp4]
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
from series_generator import _collect_layer_paths

SAVED_PRESETS_PATH = Path(__file__).resolve().parent / "saved_presets.json"
FRAME_DURATION_MS = 300
STATIC_FRAMES_PER_IMAGE = 6


def _params_from_preset(name: str) -> GlitchParams:
    """Build GlitchParams from saved preset (matches batch_gif)."""
    if not SAVED_PRESETS_PATH.exists():
        return GlitchParams()
    with open(SAVED_PRESETS_PATH, encoding="utf-8") as f:
        saved = json.load(f)
    p = saved.get(name, {})
    valid = set(GlitchParams.__dataclass_fields__)
    filtered = {k: v for k, v in p.items() if k in valid}
    return GlitchParams(**filtered)


def main():
    parser = argparse.ArgumentParser(description="Showcase MP4: all backgrounds with gentle_pulse_common")
    parser.add_argument("-o", "--output", default="output/showcase_backgrounds.mp4", help="Output MP4 path")
    parser.add_argument("--duration", type=float, default=2.5, help="Seconds per background")
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS")
    args = parser.parse_args()

    layers = _collect_layer_paths()
    if not layers["backgrounds"]:
        print("Error: No backgrounds found in SERIES/BACKGROUNDS")
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    segment_gifs: list[Path] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="showcase_"))

    try:
        for i, bg_path in enumerate(layers["backgrounds"]):
            name = bg_path.stem
            print(f"[{i + 1}/{len(layers['backgrounds'])}] {name}...", end=" ", flush=True)

            background = PILImage.open(bg_path).convert("RGBA")
            img_bg = background.convert("RGB").resize((512, 512), PILImage.Resampling.LANCZOS)
            arr = np.array(img_bg)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=temp_dir) as t1, tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, dir=temp_dir
            ) as t2:
                cv2.imwrite(t1.name, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                cv2.imwrite(t2.name, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                base_f, alt_f = t1.name, t2.name

            gif_path = temp_dir / f"seg_{i:03d}.gif"
            params = _params_from_preset("gentle_pulse_common")
            process_images_to_gif(
                [base_f, alt_f],
                str(gif_path),
                params,
                frame_duration_ms=FRAME_DURATION_MS,
                static_frames_per_image=STATIC_FRAMES_PER_IMAGE,
                bypass_mask=True,
                mask_path=None,
                max_bytes=GIF_TARGET_BYTES,
            )

            Path(base_f).unlink(missing_ok=True)
            Path(alt_f).unlink(missing_ok=True)
            segment_gifs.append(gif_path)
            print("done")

        # Concatenate: each GIF → 2.5s clip (loop if needed) → stitch
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

        print(f"Done: {out_path} ({len(clips)} backgrounds, {args.duration}s each)")

    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    exit(main())
