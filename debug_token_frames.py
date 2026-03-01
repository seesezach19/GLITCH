"""
Debug CLI: Render one token for a chosen preset, save frames 0-11 as PNGs alongside the GIF.
Usage: python debug_token_frames.py --preset dormant_core_common [--token-id 42] [-o output/]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

from series_generator import SERIES_ROOT, _composite_layers, _collect_layer_paths
from gif_processor import process_images_to_gif_boundary_aware, _load_mask_for_frame
from glitch_preset_config import PRESET_REGISTRY


def main():
    parser = argparse.ArgumentParser(description="Debug token frames for boundary-aware presets")
    parser.add_argument("--preset", default="dormant_core_common", choices=list(PRESET_REGISTRY.keys()),
                        help="Preset name")
    parser.add_argument("--token-id", type=int, default=0, help="Token ID for deterministic seed")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("--background", default=None, help="Background filename (default: first from BACKGROUNDS)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / f"debug_{args.preset}.gif"
    debug_dir = out_dir / "debug_frames"
    debug_dir.mkdir(parents=True, exist_ok=True)

    layers = _collect_layer_paths()
    if not layers["base"] or not layers["alt_bases"]:
        print("Error: Need LICHE_BASE.png and ALT_BASES")
        return 1

    bg_dir = SERIES_ROOT / "BACKGROUNDS"
    if args.background:
        bg_path = bg_dir / args.background
    else:
        bg_path = bg_dir / "THOUSAND_SWORDS_LEGENDARY.png" if (bg_dir / "THOUSAND_SWORDS_LEGENDARY.png").exists() else list(bg_dir.glob("*.png"))[0]
    if not bg_path.exists():
        print(f"Error: Background not found: {bg_path}")
        return 1

    background = PILImage.open(bg_path).convert("RGBA")
    w, h = background.size

    base = PILImage.open(layers["base"][0]).convert("RGBA")
    if base.size != (w, h):
        base = base.resize((w, h), PILImage.Resampling.LANCZOS)
    eyes = PILImage.new("RGBA", (w, h), (0, 0, 0, 0))
    if layers["eyes"]:
        eyes = PILImage.open(layers["eyes"][0]).convert("RGBA")
        if eyes.size != (w, h):
            eyes = eyes.resize((w, h), PILImage.Resampling.LANCZOS)

    img_a = _composite_layers(background, base, eyes, [])
    alt_base = PILImage.open(layers["alt_bases"][0]).convert("RGBA")
    if alt_base.size != (w, h):
        alt_base = alt_base.resize((w, h), PILImage.Resampling.LANCZOS)
    img_b = _composite_layers(background, alt_base, eyes, [])

    img_a = img_a.resize((512, 512), PILImage.Resampling.LANCZOS)
    img_b = img_b.resize((512, 512), PILImage.Resampling.LANCZOS)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t1, tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t2:
        cv2.imwrite(t1.name, cv2.cvtColor(np.array(img_a), cv2.COLOR_RGB2BGR))
        cv2.imwrite(t2.name, cv2.cvtColor(np.array(img_b), cv2.COLOR_RGB2BGR))
        base_path, alt_path = t1.name, t2.name

    mask_p = SERIES_ROOT / "LICHE_MASK.png"
    if not mask_p.exists():
        print("Error: LICHE_MASK.png not found")
        return 1

    print(f"Generating 12 frames with preset '{args.preset}', token_id={args.token_id}")
    process_images_to_gif_boundary_aware(
        [base_path, alt_path],
        str(gif_path),
        args.preset,
        token_seed=args.token_id,
        frame_duration_ms=50,
        static_frames_per_image=6,
        mask_path=str(mask_p),
        save_debug_frames=True,
    )

    Path(base_path).unlink(missing_ok=True)
    Path(alt_path).unlink(missing_ok=True)

    print(f"GIF: {gif_path}")
    print(f"Debug frames: {debug_dir}/frame_00.png .. frame_11.png")
    return 0


if __name__ == "__main__":
    exit(main())
