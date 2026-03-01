"""
Liche - Rarity-based glitch presets for Series Generator.
Edit these to fine-tune the look per rarity tier.
"""

import random
from glitch_processor import GlitchParams

# Rarity → preset (GlitchParams-compatible dict). Edit values to fine-tune.
# Cleared for redo — add your effects here.
RARITY_PRESETS = {
    "common": {},
    "uncommon": {},
    "rare": {},
    "super_rare": {},
    "legendary": {},
}

# Named presets for specific vibes (used by roll_preset_rarity when saved)
NAMED_PRESETS = {}


# Random preset weights: rarity -> probability (must sum to 100)
RANDOM_PRESET_WEIGHTS = {
    "common": 55,
    "uncommon": 25,
    "rare": 12,
    "legendary": 8,
}


def get_preset_for_rarity(rarity: str) -> GlitchParams:
    """Return GlitchParams for the given rarity tier."""
    preset = RARITY_PRESETS.get(rarity, RARITY_PRESETS["common"])
    valid = {f for f in GlitchParams.__dataclass_fields__}
    filtered = {k: v for k, v in preset.items() if k in valid}
    return GlitchParams(**filtered)


def roll_preset_rarity(seed: int | None = None) -> str:
    """Roll preset rarity by RANDOM_PRESET_WEIGHTS (55% common, 25% uncommon, 12% rare, 8% legendary)."""
    if seed is not None:
        random.seed(seed)
    rarities = list(RANDOM_PRESET_WEIGHTS.keys())
    weights = list(RANDOM_PRESET_WEIGHTS.values())
    return random.choices(rarities, weights=weights, k=1)[0]
