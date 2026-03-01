"""
Liche Series - Random NFT Layer Generator
Combines layers from series/ folder into random compositions.
Supports rarity via: 1) rarity_config.json, 2) filename suffix _common, _rare, etc.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image

SERIES_ROOT = Path(__file__).resolve().parent / "SERIES"
RARITY_CONFIG_PATH = SERIES_ROOT / "rarity_config.json"

# Mask for protected region (255 = protect, 0 = glitch). Check common names.
MASK_NAMES = ("LICHE_MASK.png", "mask.png", "MASK.png", "Mask.png", "protect.png", "PROTECT.png")


def get_series_mask_path() -> Path | None:
    """Return path to series mask if it exists, else None."""
    for name in MASK_NAMES:
        p = SERIES_ROOT / name
        if p.exists():
            return p
    return None

# Accessory layer order (first = bottom). Cracks under hats.
ACCESSORY_LAYER_ORDER = ("cigarette", "cracks", "hats", "sunglasses", "teeth")

# Accessories that render behind the base layer (file stems, case-insensitive)
BEHIND_BASE_ACCESSORIES = frozenset({"ripping_void"})

# Rarity weights (higher = more likely). No suffix = rare.
RARITY_WEIGHTS = {
    "common": 19,
    "uncommon": 10,
    "rare": 5,
    "super_rare": 2,
    "legendary": 1,
}


def _load_rarity_config() -> dict[str, str]:
    """Load rarity overrides from rarity_config.json. Keys: path relative to series (e.g. BACKGROUNDS/foo.png)."""
    if not RARITY_CONFIG_PATH.exists():
        return {}
    try:
        with open(RARITY_CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return {k.replace("\\", "/"): str(v).lower() for k, v in data.items()}
    except (json.JSONDecodeError, OSError):
        return {}


def _parse_rarity(path: Path) -> tuple[str, int]:
    """Parse rarity: 1) rarity_config.json, 2) filename suffix, 3) default rare."""
    try:
        rel = path.relative_to(SERIES_ROOT)
    except ValueError:
        rel = path
    key = str(rel).replace("\\", "/")
    config = _load_rarity_config()
    if key in config:
        tier = config[key]
        if tier in RARITY_WEIGHTS:
            return tier, RARITY_WEIGHTS[tier]
    name = path.stem.lower()
    for tier in ("legendary", "super_rare", "rare", "uncommon", "common"):
        if name.endswith("_" + tier) or name.endswith("-" + tier):
            return tier, RARITY_WEIGHTS[tier]
    return "rare", RARITY_WEIGHTS["rare"]


def _weighted_choice(paths: list[Path]) -> tuple[Path, str]:
    """Pick one path by rarity weight. Returns (path, rarity_tier)."""
    if not paths:
        raise ValueError("Empty paths")
    if len(paths) == 1:
        p = paths[0]
        return p, _parse_rarity(p)[0]
    items = [(_parse_rarity(p), p) for p in paths]
    weights = [w for (_, w), _ in items]
    tiers = [t for (t, _), _ in items]
    idx = random.choices(range(len(paths)), weights=weights, k=1)[0]
    return paths[idx], tiers[idx]


def _collect_layer_paths() -> dict:
    """Discover all layer files. Returns dict of category -> list of Paths."""
    root = SERIES_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Series folder not found: {root}")

    layers = {
        "base": [],
        "alt_bases": [],
        "backgrounds": [],
        "eyes": [],
        "cigarette": [],
        "hats": [],
        "sunglasses": [],
        "teeth": [],
    }

    # Base
    base_file = root / "LICHE_BASE.png"
    if base_file.exists():
        layers["base"].append(base_file)

    # Alt bases (flat list of all variants)
    alt_dir = root / "ALT_BASES"
    if alt_dir.exists():
        for color_dir in alt_dir.iterdir():
            if color_dir.is_dir():
                for f in color_dir.glob("*.png"):
                    layers["alt_bases"].append(f)

    # Backgrounds
    bg_dir = root / "BACKGROUNDS"
    if bg_dir.exists():
        layers["backgrounds"] = sorted(bg_dir.glob("*.png"))

    # Eyes
    eyes_dir = root / "EYES"
    if eyes_dir.exists():
        layers["eyes"] = sorted(eyes_dir.glob("*.png"))

    # Accessories — auto-discover all subfolders
    acc_root = root / "ACCESSORIES"
    if acc_root.exists():
        for subdir in sorted(acc_root.iterdir()):
            if subdir.is_dir():
                key = subdir.name.lower()
                layers[key] = sorted(subdir.glob("*.png"))

    return layers


def _composite_layers(
    background: Image.Image,
    base: Image.Image,
    eyes: Image.Image,
    accessories: list[Image.Image],
    accessories_behind_base: list[Image.Image] | None = None,
) -> Image.Image:
    """Composite layers in order. All must be same size; uses alpha blending.
    accessories_behind_base: rendered after background, before base.
    accessories: rendered after base and eyes.
    """
    w, h = background.size
    result = background.convert("RGBA")
    behind = accessories_behind_base or []

    for layer in behind + [base, eyes] + accessories:
        if layer is None:
            continue
        layer = layer.convert("RGBA")
        if layer.size != (w, h):
            layer = layer.resize((w, h), Image.Resampling.LANCZOS)
        result = Image.alpha_composite(result, layer)

    return result.convert("RGB")


def generate_pair(
    accessory_chance: float = 0.02,
    eyes_chance: float = 0.1,
    seed: int | None = None,
) -> tuple[Image.Image, Image.Image, dict]:
    """
    Generate a pair: base + attributes, and alt_base + same attributes.
    Returns (img_base, img_alt, metadata).
    accessory_chance: per-category chance to get any accessory (default 2%).
    eyes_chance: chance to get eyes (default 10%); otherwise transparent.
    """
    if seed is not None:
        random.seed(seed)

    layers = _collect_layer_paths()
    if not layers["base"]:
        raise ValueError("No base found (LICHE_BASE.png)")
    if not layers["alt_bases"]:
        raise ValueError("No alt bases found in series/ALT_BASES")

    # Background (required) — weighted by rarity
    if not layers["backgrounds"]:
        raise ValueError("No backgrounds found in series/BACKGROUNDS")
    bg_path, bg_rarity = _weighted_choice(layers["backgrounds"])
    background = Image.open(bg_path).convert("RGBA")
    w, h = background.size

    # Eyes — 10% chance; otherwise transparent
    if layers["eyes"] and random.random() < eyes_chance:
        eyes_path, eyes_rarity = _weighted_choice(layers["eyes"])
        eyes = Image.open(eyes_path).convert("RGBA")
        if eyes.size != (w, h):
            eyes = eyes.resize((w, h), Image.Resampling.LANCZOS)
        metadata_eyes, metadata_eyes_rarity = eyes_path.name, eyes_rarity
    else:
        eyes = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        metadata_eyes, metadata_eyes_rarity = "none", "common"

    # Accessories (per category) — same for both images, weighted by rarity
    # Split into behind_base (e.g. ripping_void) and in-front
    acc_images: list[Image.Image] = []
    acc_images_behind: list[Image.Image] = []
    metadata = {
        "background": bg_path.name,
        "background_rarity": bg_rarity,
        "eyes": metadata_eyes,
        "eyes_rarity": metadata_eyes_rarity,
        "accessories": [],
        "accessory_rarities": [],
    }

    accessory_keys = [
        k for k in ACCESSORY_LAYER_ORDER
        if k in layers and layers[k] and k not in ("base", "alt_bases", "backgrounds", "eyes")
    ]
    # Include any auto-discovered categories not in the order (append at end)
    extra = [k for k in layers if k not in ("base", "alt_bases", "backgrounds", "eyes") and k not in accessory_keys and layers[k]]
    accessory_keys = list(accessory_keys) + sorted(extra)
    for key in accessory_keys:
        opts = layers[key]
        if opts and random.random() < accessory_chance:
            p, acc_rarity = _weighted_choice(opts)
            img = Image.open(p).convert("RGBA")
            if img.size != (w, h):
                img = img.resize((w, h), Image.Resampling.LANCZOS)
            behind = p.stem.lower() in BEHIND_BASE_ACCESSORIES
            if behind:
                acc_images_behind.append(img)
            else:
                acc_images.append(img)
            metadata["accessories"].append(p.name)
            metadata["accessory_rarities"].append(acc_rarity)

    # Base image
    base_path = layers["base"][0]
    base = Image.open(base_path).convert("RGBA")
    if base.size != (w, h):
        base = base.resize((w, h), Image.Resampling.LANCZOS)
    img_base = _composite_layers(background, base, eyes, acc_images, acc_images_behind)
    metadata["base"] = base_path.name

    # Alt base image (same attributes) — random selection, no rarity
    alt_path = random.choice(layers["alt_bases"])
    alt_base = Image.open(alt_path).convert("RGBA")
    if alt_base.size != (w, h):
        alt_base = alt_base.resize((w, h), Image.Resampling.LANCZOS)
    img_alt = _composite_layers(background, alt_base, eyes, acc_images, acc_images_behind)
    metadata["alt_base"] = alt_path.name

    # Overall rarity (highest tier among selected traits; alt_base excluded)
    tier_order = ("common", "uncommon", "rare", "super_rare", "legendary")
    all_rarities = [bg_rarity, metadata_eyes_rarity] + metadata["accessory_rarities"]
    metadata["overall_rarity"] = max(all_rarities, key=lambda t: tier_order.index(t))

    return img_base, img_alt, metadata


def generate_batch(
    count: int,
    output_dir: str | Path,
    accessory_chance: float = 0.02,
    eyes_chance: float = 0.1,
    seed: int | None = None,
    name_pattern: str = "liche_{:04d}",
) -> list[Path]:
    """Generate count pairs (base + alt). Returns list of output paths (2 per pair)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        random.seed(seed)

    paths: list[Path] = []
    for i in range(count):
        img_base, img_alt, _ = generate_pair(accessory_chance=accessory_chance, eyes_chance=eyes_chance)
        base_path = output_dir / f"{name_pattern.format(i + 1)}_base.png"
        alt_path = output_dir / f"{name_pattern.format(i + 1)}_alt.png"
        img_base.save(base_path)
        img_alt.save(alt_path)
        paths.extend([base_path, alt_path])

    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Liche Series Random Generator")
    parser.add_argument("-n", "--count", type=int, default=1, help="Number of pairs to generate")
    parser.add_argument("-o", "--output", default="output", help="Output path (file or directory)")
    parser.add_argument("--acc-chance", type=float, default=0.02, help="Chance per accessory category (0-1)")
    parser.add_argument("--eyes-chance", type=float, default=0.1, help="Chance to get eyes (0-1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    out = Path(args.output)
    if args.count == 1:
        img_base, img_alt, meta = generate_pair(
            accessory_chance=args.acc_chance,
            eyes_chance=args.eyes_chance,
            seed=args.seed,
        )
        if out.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            out.mkdir(parents=True, exist_ok=True)
            base_path = out / "liche_base.png"
            alt_path = out / "liche_alt.png"
        else:
            base_path = out.parent / (out.stem + "_base.png")
            alt_path = out.parent / (out.stem + "_alt.png")
        base_path.parent.mkdir(parents=True, exist_ok=True)
        img_base.save(base_path)
        img_alt.save(alt_path)
        print(f"Saved: {base_path}")
        print(f"Saved: {alt_path}")
        print("Layers:", meta)
    else:
        paths = generate_batch(
            args.count,
            out,
            accessory_chance=args.acc_chance,
            eyes_chance=args.eyes_chance,
            seed=args.seed,
        )
        print(f"Generated {len(paths)} images ({args.count} pairs) in {args.output}")
