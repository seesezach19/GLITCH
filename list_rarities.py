"""List all items with their rarities (layers + glitch presets)."""
import json
from pathlib import Path

SERIES_ROOT = Path(__file__).resolve().parent / "SERIES"
RARITY_CONFIG_PATH = SERIES_ROOT / "rarity_config.json"
SAVED_PRESETS_PATH = Path(__file__).resolve().parent / "saved_presets.json"

RARITY_WEIGHTS = {"common": 19, "uncommon": 10, "rare": 5, "super_rare": 2, "legendary": 1}


def load_saved_presets():
    if not SAVED_PRESETS_PATH.exists():
        return {}
    try:
        with open(SAVED_PRESETS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_config():
    if not RARITY_CONFIG_PATH.exists():
        return {}
    try:
        with open(RARITY_CONFIG_PATH, encoding="utf-8") as f:
            return {k.replace("\\", "/"): str(v).lower() for k, v in json.load(f).items()}
    except Exception:
        return {}


def parse_rarity(path, config):
    try:
        rel = path.relative_to(SERIES_ROOT)
    except ValueError:
        rel = path
    key = str(rel).replace("\\", "/")
    if key in config and config[key] in RARITY_WEIGHTS:
        return config[key]
    name = path.stem.lower()
    for tier in ("legendary", "super_rare", "rare", "uncommon", "common"):
        if name.endswith("_" + tier) or name.endswith("-" + tier):
            return tier
    return "rare"


def parse_preset_rarity(name: str) -> str:
    """Get rarity from preset name suffix (e.g. dormant_core_common -> common)."""
    name = name.lower()
    for tier in ("legendary", "super_rare", "rare", "uncommon", "common"):
        if name.endswith("_" + tier) or name.endswith("-" + tier):
            return tier
    return "rare"


def main():
    config = load_config()
    items = []
    alt_bases = []
    for f in sorted(SERIES_ROOT.rglob("*.png")):
        if f.is_file():
            rel = str(f.relative_to(SERIES_ROOT)).replace("\\", "/")
            if "ALT_BASES" in rel:
                alt_bases.append(rel)
                continue  # Alt bases: random selection, no rarity
            rarity = parse_rarity(f, config)
            items.append((rel, rarity, "layer"))

    # Glitch presets from glitch_presets.py
    try:
        from glitch_presets import RARITY_PRESETS
        for tier in RARITY_PRESETS:
            items.append((f"[glitch_presets] {tier}", tier, "preset"))
    except ImportError:
        pass

    # Saved glitch presets
    saved = load_saved_presets()
    for name in saved:
        rarity = parse_preset_rarity(name)
        items.append((f"[saved] {name}", rarity, "preset"))

    by_rarity = {}
    for rel, r, _ in items:
        by_rarity.setdefault(r, []).append(rel)

    for tier in ("common", "uncommon", "rare", "super_rare", "legendary"):
        if tier in by_rarity:
            print(f"\n=== {tier.upper()} ===")
            # Layers first, then presets
            layers = [x for x in sorted(by_rarity[tier]) if not x.startswith("[")]
            presets = [x for x in sorted(by_rarity[tier]) if x.startswith("[")]
            for x in layers:
                print(f"  {x}")
            if presets:
                print("  --- glitch presets ---")
                for x in presets:
                    print(f"  {x}")

    if alt_bases:
        print("\n=== ALT BASES (random selection, no rarity) ===")
        for x in sorted(alt_bases):
            print(f"  {x}")


if __name__ == "__main__":
    main()
