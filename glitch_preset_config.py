"""
LICHE NFT - Preset configuration for boundary-aware A/B swap animation.
Presets define base effects (steady) and transition hit behavior (at swap boundary only).
saved_presets.json is the master: all preset names come from there.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Swap timing: 6 frames A, 6 frames B = 12-frame loop
SWAP_PERIOD = 6
TOTAL_FRAMES = 12
FRAME_DURATION_MS = 50

SAVED_PRESETS_PATH = Path(__file__).resolve().parent / "saved_presets.json"

# Keys to exclude from base_params (not steady background effects)
_EXCLUDE_KEYS = {"color_scheme", "mask_sensitivity", "mask_soft_edge", "mask_opacity", "chaos_level"}


def _load_saved_presets() -> dict:
    """Load saved_presets.json. Returns {} if missing or invalid."""
    if not SAVED_PRESETS_PATH.exists():
        return {}
    try:
        with open(SAVED_PRESETS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _base_params_from_saved(name: str, valid_keys: set) -> dict:
    """Extract base_params from saved preset: non-zero effect values only."""
    saved = _load_saved_presets()
    if name not in saved:
        return {}
    p = saved[name]
    return {k: v for k, v in p.items() if k in valid_keys and k not in _EXCLUDE_KEYS and v != 0}


@dataclass
class PresetConfig:
    """Per-preset configuration for base effects and boundary hit."""
    name: str
    # Base effects: steady, loaded from saved_presets.json when available
    base_params: dict = field(default_factory=dict)
    # Boundary hit strength: 0..1. Commons ~0.3-0.6, rares/legendaries 0.8-1.0
    boundary_strength_min: float = 0.3
    boundary_strength_max: float = 0.6
    # Hit components: subset of ("block_tear", "rgb_shift", "vhs_tracking", "rolling_scanlines", "brightness_pulse", "abyssal_tear", "ghost_trail", "chromatic_aberration")
    hit_components: tuple = ("block_tear", "rgb_shift")
    # After-effect: 1-2 frames decay after boundary (legendary)
    after_effect_frames: int = 0
    # If True, apply effects to full image (no mask protection). Default True for all presets.
    bypass_mask: bool = True


# Preset definitions. base_params are merged with saved_presets.json at runtime.
_PRESET_DEFAULTS: dict[str, dict] = {
    "dormant_core_common": {
        "boundary_strength_min": 0.3, "boundary_strength_max": 0.5,
        "hit_components": ("block_tear", "brightness_pulse"),
        "after_effect_frames": 0,
    },
    "gentle_pulse_common": {
        "boundary_strength_min": 0.4, "boundary_strength_max": 0.6,
        "hit_components": ("rolling_scanlines", "block_tear"),
        "after_effect_frames": 0,
    },
    "chromatic_bloom_uncommon": {
        "boundary_strength_min": 0.6, "boundary_strength_max": 0.8,
        "hit_components": ("rgb_shift", "chromatic_aberration", "brightness_pulse"),
        "after_effect_frames": 0,
    },
    "scanline_whisper_uncommon": {
        "boundary_strength_min": 0.6, "boundary_strength_max": 0.8,
        "hit_components": ("vhs_tracking", "rolling_scanlines", "block_tear"),
        "after_effect_frames": 0,
    },
    "abyssal_tear_rare": {
        "boundary_strength_min": 0.8, "boundary_strength_max": 1.0,
        "hit_components": ("abyssal_tear", "block_tear", "rgb_shift"),
        "after_effect_frames": 0,
    },
    "living_abyss_legendary": {
        "boundary_strength_min": 0.9, "boundary_strength_max": 1.0,
        "hit_components": ("abyssal_tear", "block_tear", "vhs_tracking"),
        "after_effect_frames": 2,
    },
    "soul_shatter_legendary": {
        "boundary_strength_min": 0.9, "boundary_strength_max": 1.0,
        "hit_components": ("ghost_trail", "block_tear", "rgb_shift"),
        "after_effect_frames": 1,
    },
    "necrotic_flow_legendary": {
        "boundary_strength_min": 0.9, "boundary_strength_max": 1.0,
        "hit_components": ("block_tear", "rgb_shift"),
        "after_effect_frames": 1,
    },
}

# GlitchParams field names for filtering saved preset keys
_GLITCH_PARAM_KEYS = {
    "rgb_shift_intensity", "chromatic_aberration", "scanlines", "digital_noise",
    "pixelation", "datamosh", "melting", "vhs_tracking", "crt_flicker", "bitcrush",
    "tv_static", "frame_drift", "rolling_scanlines", "foreground_effect",
    "subject_invert", "subject_particles", "edge_pulse", "ghost_trail",
    "block_tear", "vignette", "bloom", "film_grain", "pixel_sort", "neon_bars",
    "edge_dissolve", "chroma_dropout",
    "flow_warp", "slit_scan", "parallax_split", "voronoi_shatter",
    "echo_crown", "strobe_phase", "frame_drop",
    "palette_cycle", "chroma_collapse", "ordered_dither",
    "sigil_ring", "phylactery_glow", "abyss_window",
    "fft_jitter", "moire_grid",
    "shader_necrotic_iridescent_flow_intensity",
    "shader_hexagonal_warp_intensity",
    "shader_caustic_flow_intensity",
    "shader_thermal_distort_intensity",
    "transition_band_wipe", "transition_diagonal_rip", "transition_slit_scan_swap",
    "transition_pixel_scramble_patch", "transition_edge_first_reveal",
    "transition_voronoi_shatter_swap", "transition_phase_offset_echo",
    "transition_palette_snap_posterize", "transition_chroma_dropout_rebound",
    "transition_scanline_gate", "transition_micro_jitter_rgb",
    "transition_noise_threshold_crossfade",
}


def _get_boundary_defaults_for_name(name: str) -> dict:
    """Return boundary/hit defaults. Use _PRESET_DEFAULTS if present, else derive from rarity suffix."""
    if name in _PRESET_DEFAULTS:
        return _PRESET_DEFAULTS[name]
    # Fallback from rarity suffix
    if name.endswith("_legendary"):
        return {"boundary_strength_min": 0.9, "boundary_strength_max": 1.0, "hit_components": ("block_tear", "rgb_shift"), "after_effect_frames": 1}
    if name.endswith("_rare"):
        return {"boundary_strength_min": 0.8, "boundary_strength_max": 1.0, "hit_components": ("block_tear", "rgb_shift"), "after_effect_frames": 0}
    if name.endswith("_uncommon"):
        return {"boundary_strength_min": 0.6, "boundary_strength_max": 0.8, "hit_components": ("block_tear", "rgb_shift"), "after_effect_frames": 0}
    # common or unknown
    return {"boundary_strength_min": 0.3, "boundary_strength_max": 0.6, "hit_components": ("block_tear", "brightness_pulse"), "after_effect_frames": 0}


def _build_preset_registry() -> dict[str, PresetConfig]:
    """Build PRESET_REGISTRY from saved_presets.json (master). All preset names come from there."""
    saved = _load_saved_presets()
    names = list(saved.keys()) if saved else list(_PRESET_DEFAULTS.keys())
    registry = {}
    for name in names:
        base_params = _base_params_from_saved(name, _GLITCH_PARAM_KEYS)
        defaults = _get_boundary_defaults_for_name(name)
        saved_p = saved.get(name, {}) if saved else {}
        bypass = saved_p.get("bypass_mask", True)
        registry[name] = PresetConfig(
            name=name,
            base_params=base_params,
            boundary_strength_min=defaults["boundary_strength_min"],
            boundary_strength_max=defaults["boundary_strength_max"],
            hit_components=defaults["hit_components"],
            after_effect_frames=defaults["after_effect_frames"],
            bypass_mask=bool(bypass),
        )
    return registry


PRESET_REGISTRY: dict[str, PresetConfig] = _build_preset_registry()


def get_preset_config(preset_name: str) -> Optional[PresetConfig]:
    """Return preset config by name, or None if not found."""
    return PRESET_REGISTRY.get(preset_name)


def get_boundary_strength(frame_idx: int, config: PresetConfig, token_seed: int = 0) -> float:
    """
    Return 0..1 strength for transition hit.
    Peak at frames 5 and 11; ramp at 4 and 10.
    """
    pos = frame_idx % SWAP_PERIOD
    if pos == SWAP_PERIOD - 1:  # frame 5 or 11 - peak
        return config.boundary_strength_max
    if pos == SWAP_PERIOD - 2:  # frame 4 or 10 - ramp
        return (config.boundary_strength_min + config.boundary_strength_max) / 2
    # After-effect decay for legendary (1-2 frames after boundary)
    if config.after_effect_frames > 0:
        # Boundary was at frame (cycle_start + SWAP_PERIOD - 1)
        cycle_start = (frame_idx // SWAP_PERIOD) * SWAP_PERIOD
        boundary_frame = cycle_start + SWAP_PERIOD - 1
        dist_from_boundary = frame_idx - boundary_frame
        if 1 <= dist_from_boundary <= config.after_effect_frames:
            decay = 1.0 - (dist_from_boundary / (config.after_effect_frames + 1))
            return (config.boundary_strength_min + (config.boundary_strength_max - config.boundary_strength_min) * decay * 0.5)
    return 0.0


def get_swap_phase_info(frame_idx: int, swap_period: int = SWAP_PERIOD) -> dict:
    """
    Return phase and boundary info for A/B swap.
    phase: 0 = A (frames 0-5), 1 = B (frames 6-11)
    is_boundary: True at frames 4, 5, 10, 11
    """
    phase = (frame_idx // swap_period) % 2
    pos_in_period = frame_idx % swap_period
    is_boundary = pos_in_period in (swap_period - 1, swap_period - 2)
    return {"phase": phase, "is_boundary": is_boundary, "pos_in_period": pos_in_period}
