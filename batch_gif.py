"""
Batch GIF Generator - Generate multiple liche GIFs from the command line.
Uses same pipeline as Photos → GIF: process_images_to_gif, 300ms per image, 6 frames per image.
Usage: python batch_gif.py -n 10 -o output [options]
Output structure:
  output/static/0001.png   - first frame (thumbnail)
  output/animated/0001.gif - full animation
  output/metadata/0001.json - name, traits, image, animation_url
"""
import argparse
import json
import random
import tempfile
from pathlib import Path

from PIL import Image as PILImage

from glitch_processor import GlitchParams
from gif_processor import process_images_to_gif, GIF_TARGET_BYTES
from glitch_presets import roll_preset_rarity
from series_generator import generate_pair, SERIES_ROOT

SAVED_PRESETS_PATH = Path(__file__).resolve().parent / "saved_presets.json"
FRAME_DURATION_MS = 300
STATIC_FRAMES_PER_IMAGE = 6


def _load_saved_presets() -> dict:
    if not SAVED_PRESETS_PATH.exists():
        return {}
    try:
        with open(SAVED_PRESETS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _params_from_preset(name: str) -> GlitchParams:
    """Build GlitchParams from saved preset. Filters to valid fields only."""
    saved = _load_saved_presets()
    p = saved.get(name, {})
    valid = set(GlitchParams.__dataclass_fields__)
    filtered = {k: v for k, v in p.items() if k in valid}
    return GlitchParams(**filtered)


def _get_random_preset(seed: int) -> tuple[GlitchParams, str]:
    """Pick preset by rarity (roll) then random from saved_presets matching that rarity."""
    saved = _load_saved_presets()
    if not saved:
        return GlitchParams(), "default"
    rarity = roll_preset_rarity(seed=seed)
    suffix = "_" + rarity
    matches = [n for n in saved if n.endswith(suffix)]
    if not matches:
        matches = list(saved.keys())
    random.seed(seed + 1)
    name = random.choice(matches)
    return _params_from_preset(name), name


def build_nft_metadata(
    token_id: str,
    meta: dict,
    image_path: str,
    animation_url: str,
) -> dict:
    """Build NFT metadata with name, traits, image, animation_url."""
    traits = []
    if meta.get("background"):
        traits.append({"trait_type": "Background", "value": meta["background"].replace(".png", "")})
    if meta.get("background_rarity"):
        traits.append({"trait_type": "Background Rarity", "value": meta["background_rarity"].title()})
    if meta.get("eyes") and meta["eyes"] != "none":
        traits.append({"trait_type": "Eyes", "value": meta["eyes"].replace(".png", "")})
    if meta.get("eyes_rarity"):
        traits.append({"trait_type": "Eyes Rarity", "value": meta["eyes_rarity"].title()})
    for acc, rar in zip(meta.get("accessories", []), meta.get("accessory_rarities", [])):
        traits.append({"trait_type": "Accessory", "value": acc.replace(".png", ""), "rarity": rar})
    if meta.get("base"):
        traits.append({"trait_type": "Base", "value": meta["base"].replace(".png", "")})
    if meta.get("alt_base"):
        traits.append({"trait_type": "Alt Base", "value": meta["alt_base"].replace(".png", "")})
    if meta.get("overall_rarity"):
        traits.append({"trait_type": "Overall Rarity", "value": meta["overall_rarity"].replace("_", " ").title()})
    if meta.get("preset_rarity"):
        traits.append({"trait_type": "Glitch Preset", "value": meta["preset_rarity"]})

    return {
        "name": f"Liche #{token_id}",
        "traits": traits,
        "attributes": traits,  # NFT standard alias
        "image": image_path,
        "animation_url": animation_url,
        "seed": meta.get("seed"),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch generate liche GIFs")
    parser.add_argument("-n", "--count", type=int, default=10, help="Number of GIFs to generate")
    parser.add_argument("-o", "--output", default="output", help="Output directory (creates static/, animated/, metadata/)")
    parser.add_argument("--base-url", default="", help="Base URL for image/animation_url (e.g. https://example.com/)")
    parser.add_argument("--metadata-only", action="store_true", help="Generate metadata only (no GIFs/PNGs)")
    parser.add_argument("--acc-chance", type=float, default=0.02)
    parser.add_argument("--eyes-chance", type=float, default=0.1)
    parser.add_argument("--preset", choices=["traits", "random"], default="random")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-mask", action="store_true", help="Use auto-detect mask from each frame to protect subject")
    args = parser.parse_args()

    if not SERIES_ROOT.exists():
        print(f"Error: Series folder not found: {SERIES_ROOT}")
        return 1

    out_dir = Path(args.output)
    static_dir = out_dir / "static"
    animated_dir = out_dir / "animated"
    metadata_dir = out_dir / "metadata"
    base_url = (args.base_url or "").rstrip("/")
    if base_url:
        base_url += "/"

    static_dir.mkdir(parents=True, exist_ok=True)
    animated_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        token_id = f"{i+1:04d}"
        seed = (args.seed or 0) + i
        print(f"[{i+1}/{args.count}] Generating...", end=" ", flush=True)
        img_base, img_alt, meta = generate_pair(
            accessory_chance=args.acc_chance,
            eyes_chance=args.eyes_chance,
            seed=seed,
        )
        params, preset_name = _get_random_preset(seed)

        meta["preset_rarity"] = preset_name
        meta["seed"] = seed

        image_path = f"{base_url}static/{token_id}.png"
        animation_url = f"{base_url}animated/{token_id}.gif"
        nft_meta = build_nft_metadata(token_id, meta, image_path, animation_url)

        if not args.metadata_only:
            img_base = img_base.resize((512, 512), PILImage.Resampling.LANCZOS)
            img_alt = img_alt.resize((512, 512), PILImage.Resampling.LANCZOS)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t1, tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t2:
                img_base.save(t1.name)
                img_alt.save(t2.name)
                base_path, alt_path = t1.name, t2.name

            gif_path = animated_dir / f"{token_id}.gif"
            # When --use-mask: bypass mask file, always use auto-detect from frame
            process_images_to_gif(
                [base_path, alt_path],
                str(gif_path),
                params,
                frame_duration_ms=FRAME_DURATION_MS,
                static_frames_per_image=STATIC_FRAMES_PER_IMAGE,
                bypass_mask=not args.use_mask,
                mask_path=None,
                max_bytes=GIF_TARGET_BYTES,
            )
            Path(base_path).unlink(missing_ok=True)
            Path(alt_path).unlink(missing_ok=True)

            # Extract first frame as PNG thumbnail
            with PILImage.open(gif_path) as gif:
                first_frame = gif.convert("RGB")
            first_frame.save(static_dir / f"{token_id}.png", "PNG")

        with open(metadata_dir / f"{token_id}.json", "w", encoding="utf-8") as f:
            json.dump(nft_meta, f, indent=2)
        print(f"-> {token_id}")

    if args.metadata_only:
        print(f"\nDone. metadata/ ({args.count} JSON files) in {out_dir}")
    else:
        print(f"\nDone. {args.count} items in {out_dir} (static/, animated/, metadata/)")
    return 0


if __name__ == "__main__":
    exit(main())
