"""
Generate a glitched GIF with the Thousand Swords background.
Uses boundary-aware A/B swap (6 frames each, transition hit at boundary).
Usage: python generate_thousand_swords_gif.py [-o output.gif]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

from series_generator import SERIES_ROOT, _composite_layers, _collect_layer_paths
from gif_processor import process_images_to_gif_boundary_aware
from glitch_preset_config import PRESET_REGISTRY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="output/thousand_swords.gif", help="Output GIF path")
    parser.add_argument("--preset", default="living_abyss_legendary", choices=list(PRESET_REGISTRY.keys()),
                        help="Preset name")
    parser.add_argument("--token-id", type=int, default=0, help="Token ID for deterministic seed")
    args = parser.parse_args()

    # Load Thousand Swords background
    bg_path = SERIES_ROOT / "BACKGROUNDS" / "THOUSAND_SWORDS_LEGENDARY.png"
    if not bg_path.exists():
        print(f"Error: {bg_path} not found")
        return 1

    layers = _collect_layer_paths()
    if not layers["base"]:
        print("Error: No LICHE_BASE.png found")
        return 1
    if not layers["alt_bases"]:
        print("Error: No alt bases found")
        return 1

    background = PILImage.open(bg_path).convert("RGBA")
    w, h = background.size

    # Base + eyes (no accessories for simplicity)
    base = PILImage.open(layers["base"][0]).convert("RGBA")
    if base.size != (w, h):
        base = base.resize((w, h), PILImage.Resampling.LANCZOS)

    eyes = PILImage.new("RGBA", (w, h), (0, 0, 0, 0))
    if layers["eyes"]:
        eyes = PILImage.open(layers["eyes"][0]).convert("RGBA")
        if eyes.size != (w, h):
            eyes = eyes.resize((w, h), PILImage.Resampling.LANCZOS)

    img_base = _composite_layers(background, base, eyes, [])

    # Alt base (first alt)
    alt_base = PILImage.open(layers["alt_bases"][0]).convert("RGBA")
    if alt_base.size != (w, h):
        alt_base = alt_base.resize((w, h), PILImage.Resampling.LANCZOS)
    img_alt = _composite_layers(background, alt_base, eyes, [])

    # Resize to 512x512 for GIF
    img_base = img_base.resize((512, 512), PILImage.Resampling.LANCZOS)
    img_alt = img_alt.resize((512, 512), PILImage.Resampling.LANCZOS)

    # Save temp images for gif processor (expects BGR paths)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t1, tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t2:
        cv2.imwrite(t1.name, cv2.cvtColor(np.array(img_base), cv2.COLOR_RGB2BGR))
        cv2.imwrite(t2.name, cv2.cvtColor(np.array(img_alt), cv2.COLOR_RGB2BGR))
        base_path, alt_path = t1.name, t2.name

    mask_p = SERIES_ROOT / "LICHE_MASK.png"
    if not mask_p.exists():
        print("Error: LICHE_MASK.png not found")
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating GIF with Thousand Swords background -> {out_path} (preset={args.preset})")
    process_images_to_gif_boundary_aware(
        [base_path, alt_path],
        str(out_path),
        args.preset,
        token_seed=args.token_id,
        frame_duration_ms=50,
        static_frames_per_image=6,
        mask_path=str(mask_p),
    )

    Path(base_path).unlink(missing_ok=True)
    Path(alt_path).unlink(missing_ok=True)
    print(f"Done: {out_path}")
    return 0


if __name__ == "__main__":
    exit(main())
