"""
Liche
NFT series generator + glitched GIF from photos.
"""

import json
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image as PILImage

from glitch_processor import GlitchParams
from gif_processor import process_images_to_gif, convert_gif_to_mp4, GIF_TARGET_BYTES
from glitch_presets import get_preset_for_rarity, roll_preset_rarity
from series_generator import generate_pair, generate_batch, SERIES_ROOT
from gif_overlay_utils import overlay_gif_on_image

SERIES_AVAILABLE = SERIES_ROOT.exists()

# =============================================================================
# Page Config & Custom CSS - Dark Cyberpunk / Black Metal Aesthetic
# =============================================================================

st.set_page_config(
    page_title="Liche",
    page_icon="☠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CYBERPUNK_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

:root {
    --bg-dark: #0a0a0a;
    --neon-cyan: #00ffff;
    --blood-red: #ff0040;
    --dark-gray: #1a1a1a;
}

.stApp {
    background: linear-gradient(180deg, #0a0a0a 0%, #0d0d0d 100%);
}

h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--neon-cyan) !important;
    text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
}

.glitch-title {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 900 !important;
    font-size: 2.5rem !important;
    color: var(--neon-cyan) !important;
    text-shadow: 
        2px 2px 0 var(--blood-red),
        -2px -2px 0 var(--blood-red),
        0 0 20px rgba(0, 255, 255, 0.8);
    animation: glitch-pulse 2s ease-in-out infinite;
}

@keyframes glitch-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.9; }
}

.main .block-container {
    padding: 1rem 2rem;
    max-width: 100%;
}

.stButton > button {
    font-family: 'Share Tech Mono', monospace !important;
    background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%) !important;
    color: var(--neon-cyan) !important;
    border: 2px solid var(--neon-cyan) !important;
    border-radius: 4px !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.5) !important;
    background: rgba(0, 255, 255, 0.1) !important;
}

.big-generate-btn > button {
    font-size: 1.2rem !important;
    padding: 1rem 2rem !important;
    border: 3px solid var(--blood-red) !important;
    color: var(--blood-red) !important;
    box-shadow: 0 0 30px rgba(255, 0, 64, 0.3) !important;
}

.big-generate-btn > button:hover {
    box-shadow: 0 0 40px rgba(255, 0, 64, 0.6) !important;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d0d 0%, #0a0a0a 100%);
    border-right: 1px solid rgba(0, 255, 255, 0.2);
}

.stSlider > div > div {
    background: linear-gradient(90deg, var(--neon-cyan), var(--blood-red)) !important;
}

.stSlider > div > div > div {
    background: var(--neon-cyan) !important;
}

div[data-baseweb="select"] {
    background: var(--dark-gray) !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, var(--blood-red), var(--neon-cyan)) !important;
}

.gallery-card {
    background: var(--dark-gray);
    border: 1px solid rgba(0, 255, 255, 0.3);
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.footer-text {
    font-family: 'Share Tech Mono', monospace;
    color: #666;
    font-size: 0.85rem;
    text-align: center;
    margin-top: 3rem;
    padding: 1rem;
    border-top: 1px solid rgba(0, 255, 255, 0.1);
}

.stAlert {
    background: rgba(255, 0, 64, 0.1) !important;
    border: 1px solid var(--blood-red) !important;
}
"""

st.markdown(f"<style>{CYBERPUNK_CSS}</style>", unsafe_allow_html=True)


# =============================================================================
# Sidebar - Glitch Controls (for Photos → GIF)
# =============================================================================

SAVED_PRESETS_PATH = Path(__file__).resolve().parent / "saved_presets.json"

BUILTIN_PRESETS = {"custom": None}


def load_saved_presets() -> dict:
    """Load user-saved presets from JSON file."""
    if not SAVED_PRESETS_PATH.exists():
        return {}
    try:
        with open(SAVED_PRESETS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_preset(name: str, params: dict) -> None:
    """Save a preset to the JSON file."""
    saved = load_saved_presets()
    saved[name] = params
    with open(SAVED_PRESETS_PATH, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2)


def delete_preset(name: str) -> None:
    """Remove a saved preset."""
    saved = load_saved_presets()
    if name in saved:
        del saved[name]
        with open(SAVED_PRESETS_PATH, "w", encoding="utf-8") as f:
            json.dump(saved, f, indent=2)


def get_random_preset_for_rarity(rarity: str, seed: int | None = None) -> tuple[GlitchParams, str]:
    """Pick a random saved preset matching rarity (name ends with _rarity). Fallback to glitch_presets."""
    if seed is not None:
        random.seed(seed)
    saved = load_saved_presets()
    suffix = "_" + rarity
    matches = [name for name in saved if name.endswith(suffix)]
    if matches:
        name = random.choice(matches)
        valid = set(GlitchParams.__dataclass_fields__)
        filtered = {k: v for k, v in saved[name].items() if k in valid}
        return GlitchParams(**filtered), name
    return get_preset_for_rarity(rarity), rarity


def render_sidebar():
    """Glitch effect controls for Photos → GIF mode."""
    st.sidebar.markdown("## ⚙️ Glitch Controls")
    st.sidebar.markdown("*For Photos → GIF mode*")
    st.sidebar.markdown("---")

    all_presets = {**BUILTIN_PRESETS, **load_saved_presets()}
    preset_choice = st.sidebar.selectbox(
        "Preset",
        options=list(all_presets.keys()),
        format_func=lambda x: "Custom" if x == "custom" else x.replace("_", " ").title(),
        key="sb_preset",
    )
    preset = all_presets.get(preset_choice) or {}

    chaos = st.sidebar.slider("Chaos Level", 0.0, 1.0, 0.0, 0.1, key="sb_chaos")
    mask_sensitivity = st.sidebar.slider("Mask sensitivity", 0.0, 1.0, 0.0, 0.05, key="sb_mask_sens")
    mask_soft = st.sidebar.slider("Soft edge (px)", 0, 25, 0, 1, key="sb_mask_soft")
    mask_opacity = st.sidebar.slider(
        "Mask opacity",
        0.0, 1.0, 1.0, 0.05,
        key="sb_mask_opacity",
        help="1=subject protected, 0=effects on both subject & background",
    )

    st.sidebar.markdown("### Core")
    rgb_shift = st.sidebar.slider("RGB Shift", 0, 10, 0, key="sb_rgb")
    chromatic = st.sidebar.slider("Chromatic Aberration", 0, 10, 0, key="sb_chromatic")
    scanlines = st.sidebar.slider("Scanlines", 0, 10, 0, key="sb_scanlines")
    noise = st.sidebar.slider("Digital Noise", 0, 10, 0, key="sb_noise")
    pixelation = st.sidebar.slider("Pixelation", 0, 10, 0, key="sb_pixelation")
    datamosh = st.sidebar.slider("Datamosh", 0, 10, 0, key="sb_datamosh")
    melting = st.sidebar.slider("Melting", 0, 10, 0, key="sb_melting")

    st.sidebar.markdown("### Spatial / Geometry")
    flow_warp = st.sidebar.slider("Flow Warp", 0, 10, 0, key="sb_flow_warp", help="Displacement field - liquid space-time distortion")
    slit_scan = st.sidebar.slider("Slit-Scan", 0, 10, 0, key="sb_slit_scan", help="Temporal smearing from past frames")
    parallax_split = st.sidebar.slider("Parallax Split", 0, 10, 0, key="sb_parallax_split", help="Depth layers offset - soul leaving body")
    voronoi_shatter = st.sidebar.slider("Voronoi Shatter", 0, 10, 0, key="sb_voronoi_shatter", help="Tessellated crystalline fracture")

    st.sidebar.markdown("### Temporal")
    echo_crown = st.sidebar.slider("Echo Crown", 0, 10, 0, key="sb_echo_crown", help="Directional echoes with decay mask")
    strobe_phase = st.sidebar.slider("Strobe Phase", 0, 10, 0, key="sb_strobe_phase", help="Luminance posterize snap at boundary")
    frame_drop = st.sidebar.slider("Frame Drop", 0, 10, 0, key="sb_frame_drop", help="Repeat frame + strong hit - VHS authentic")

    st.sidebar.markdown("### Palette / Color (advanced)")
    palette_cycle = st.sidebar.slider("Palette Cycle", 0, 10, 0, key="sb_palette_cycle", help="Indexed hue rotation")
    chroma_collapse = st.sidebar.slider("Chroma Collapse", 0, 10, 0, key="sb_chroma_collapse", help="Monochrome → oversaturate rebound - soul drain")
    ordered_dither = st.sidebar.slider("Ordered Dither", 0, 10, 0, key="sb_ordered_dither", help="Bayer dither overlay - ritual encoded")

    st.sidebar.markdown("### Occult Tech")
    sigil_ring = st.sidebar.slider("Sigil Ring", 0, 10, 0, key="sb_sigil_ring", help="Broken ring + tick marks")
    phylactery_glow = st.sidebar.slider("Phylactery Glow", 0, 10, 0, key="sb_phylactery_glow", help="Floating orb pulse, chroma bleed")
    abyss_window = st.sidebar.slider("Abyss Window", 0, 10, 0, key="sb_abyss_window", help="Portal interior - void/starfield band")

    st.sidebar.markdown("### Mathy Glitches")
    fft_jitter = st.sidebar.slider("FFT Jitter", 0, 10, 0, key="sb_fft_jitter", help="Phase perturbation - surreal shimmer")
    moire_grid = st.sidebar.slider("Moiré Grid", 0, 10, 0, key="sb_moire_grid", help="Angled drifting grid - cursed signal")

    st.sidebar.markdown("### GLSL Shaders")
    shader_necrotic = st.sidebar.slider("Necrotic Iridescent Flow", 0, 10, 0, key="sb_shader_necrotic", help="Organic liquid warp, purples→greens→void black, feedback flow")
    shader_hexagonal = st.sidebar.slider("Hexagonal Warp", 0, 10, 0, key="sb_shader_hexagonal", help="Honeycomb cells with liquid distortion inside each cell")
    shader_caustic = st.sidebar.slider("Caustic Flow", 0, 10, 0, key="sb_shader_caustic", help="Underwater/glass light caustics, refraction ripples")
    shader_thermal = st.sidebar.slider("Thermal Distort", 0, 10, 0, key="sb_shader_thermal", help="Heat-wave shimmer, hot pavement liquid effect")

    st.sidebar.markdown("### Transitions (A↔B boundary)")
    with st.sidebar.expander("Transition effects", expanded=False):
        trans_band_wipe = st.slider("Band Wipe", 0, 10, 0, key="sb_trans_band", help="Horizontal bands with jagged edges")
        trans_diagonal_rip = st.slider("Diagonal Rip", 0, 10, 0, key="sb_trans_diag", help="Diagonal band + rim glow")
        trans_slit_scan = st.slider("Slit-Scan Swap", 0, 10, 0, key="sb_trans_slit", help="Vertical slices A→B")
        trans_pixel_patch = st.slider("Pixel Scramble Patch", 0, 10, 0, key="sb_trans_patch", help="Localized blocky scramble")
        trans_edge_reveal = st.slider("Edge-First Reveal", 0, 10, 0, key="sb_trans_edge", help="B along silhouette, then fill")
        trans_voronoi = st.slider("Voronoi Shatter Swap", 0, 10, 0, key="sb_trans_voronoi", help="Crystalline fracture")
        trans_phase_echo = st.slider("Phase Offset Echo", 0, 10, 0, key="sb_trans_phase", help="Double exposure two skulls")
        trans_palette_snap = st.slider("Palette Snap", 0, 10, 0, key="sb_trans_palette", help="Posterize then snap to B")
        trans_chroma_rebound = st.slider("Chroma Dropout Rebound", 0, 10, 0, key="sb_trans_chroma", help="Monochrome → oversat")
        trans_scanline_gate = st.slider("Scanline Gate", 0, 10, 0, key="sb_trans_scanline", help="Rolling scanline A/B split")
        trans_micro_jitter = st.slider("Micro Jitter RGB", 0, 10, 0, key="sb_trans_jitter", help="1-2px jitter + RGB snap")
        trans_noise_crossfade = st.slider("Noise Threshold Crossfade", 0, 10, 0, key="sb_trans_noise", help="Seeded noise reveal")

    st.sidebar.markdown("### Glitch Art")
    pixel_sort = st.sidebar.slider("Pixel Sort", 0, 10, 0, key="sb_pixel_sort")
    neon_bars = st.sidebar.slider("Neon Bars", 0, 10, 0, key="sb_neon_bars")
    edge_dissolve = st.sidebar.slider("Edge Dissolve", 0, 10, 0, key="sb_edge_dissolve")
    chroma_dropout = st.sidebar.slider("Chroma Dropout", 0, 10, 0, key="sb_chroma_dropout", help="YUV: grey drop → oversaturated rebound")
    ghost_trail = st.sidebar.slider("Ghost Trail", 0, 10, 0, key="sb_ghost_trail")
    block_tear = st.sidebar.slider("Block Tear", 0, 10, 0, key="sb_block_tear")

    st.sidebar.markdown("### Atmosphere")
    tv_static = st.sidebar.slider("TV Static", 0, 10, 0, key="sb_tv_static")
    film_grain = st.sidebar.slider("Film Grain", 0, 10, 0, key="sb_film_grain")
    vignette = st.sidebar.slider("Vignette", 0, 10, 0, key="sb_vignette")
    bloom = st.sidebar.slider("Bloom", 0, 10, 0, key="sb_bloom")
    vhs = st.sidebar.slider("VHS Tracking", 0, 10, 0, key="sb_vhs")
    crt = st.sidebar.slider("CRT Flicker", 0, 10, 0, key="sb_crt")
    bitcrush = st.sidebar.slider("Bitcrush", 0, 10, 0, key="sb_bitcrush")

    st.sidebar.markdown("### Subject")
    fg_effect = st.sidebar.slider("Foreground effect", 0, 10, 0, key="sb_fg_effect")
    subject_invert = st.sidebar.slider("Subject invert", 0, 10, 0, key="sb_subject_invert")
    subject_particles = st.sidebar.slider("Subject particles", 0, 10, 0, key="sb_subject_particles")
    edge_pulse = st.sidebar.slider("Edge pulse", 0, 10, 0, key="sb_edge_pulse")

    st.sidebar.markdown("### Color")
    color_scheme = st.sidebar.selectbox(
        "Color tint",
        options=["default", "cold", "warm", "neon", "inverted", "vhs", "sepia", "cyan_magenta", "blood"],
        index=0,
        key="sb_color",
    )

    frame_drift = st.sidebar.slider("Frame Drift", 0, 10, 0, key="sb_frame_drift")
    rolling = st.sidebar.slider("Rolling Scanlines", 0, 10, 0, key="sb_rolling")

    params_dict = {
        "rgb_shift_intensity": rgb_shift,
        "chromatic_aberration": chromatic,
        "scanlines": scanlines,
        "digital_noise": noise,
        "pixelation": pixelation,
        "datamosh": datamosh,
        "melting": melting,
        "vhs_tracking": vhs,
        "crt_flicker": crt,
        "bitcrush": bitcrush,
        "tv_static": tv_static,
        "frame_drift": frame_drift,
        "rolling_scanlines": rolling,
        "color_scheme": color_scheme,
        "mask_sensitivity": mask_sensitivity,
        "mask_soft_edge": mask_soft,
        "mask_opacity": mask_opacity,
        "foreground_effect": fg_effect,
        "subject_invert": subject_invert,
        "subject_particles": subject_particles,
        "edge_pulse": edge_pulse,
        "chaos_level": chaos,
        "ghost_trail": ghost_trail,
        "block_tear": block_tear,
        "vignette": vignette,
        "bloom": bloom,
        "film_grain": film_grain,
        "pixel_sort": pixel_sort,
        "neon_bars": neon_bars,
        "edge_dissolve": edge_dissolve,
        "chroma_dropout": chroma_dropout,
        "flow_warp": flow_warp,
        "slit_scan": slit_scan,
        "parallax_split": parallax_split,
        "voronoi_shatter": voronoi_shatter,
        "echo_crown": echo_crown,
        "strobe_phase": strobe_phase,
        "frame_drop": frame_drop,
        "palette_cycle": palette_cycle,
        "chroma_collapse": chroma_collapse,
        "ordered_dither": ordered_dither,
        "sigil_ring": sigil_ring,
        "phylactery_glow": phylactery_glow,
        "abyss_window": abyss_window,
        "fft_jitter": fft_jitter,
        "moire_grid": moire_grid,
        "shader_necrotic_iridescent_flow_intensity": shader_necrotic,
        "shader_hexagonal_warp_intensity": shader_hexagonal,
        "shader_caustic_flow_intensity": shader_caustic,
        "shader_thermal_distort_intensity": shader_thermal,
        "transition_band_wipe": trans_band_wipe,
        "transition_diagonal_rip": trans_diagonal_rip,
        "transition_slit_scan_swap": trans_slit_scan,
        "transition_pixel_scramble_patch": trans_pixel_patch,
        "transition_edge_first_reveal": trans_edge_reveal,
        "transition_voronoi_shatter_swap": trans_voronoi,
        "transition_phase_offset_echo": trans_phase_echo,
        "transition_palette_snap_posterize": trans_palette_snap,
        "transition_chroma_dropout_rebound": trans_chroma_rebound,
        "transition_scanline_gate": trans_scanline_gate,
        "transition_micro_jitter_rgb": trans_micro_jitter,
        "transition_noise_threshold_crossfade": trans_noise_crossfade,
    }
    if preset:
        params_dict.update({k: v for k, v in preset.items() if k in params_dict})

    valid_keys = set(GlitchParams.__dataclass_fields__)
    params_dict = {k: v for k, v in params_dict.items() if k in valid_keys}

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Save / Load")
    save_name = st.sidebar.text_input("Preset name", placeholder="my_custom_preset", key="sb_save_name")
    col_save, col_del = st.sidebar.columns(2)
    with col_save:
        if st.button("Save current", key="sb_save_btn"):
            if save_name and save_name.strip():
                name = save_name.strip().replace(" ", "_").lower()
                save_preset(name, params_dict)
                st.sidebar.success(f"Saved as **{name}**")
            else:
                st.sidebar.warning("Enter a preset name")
    with col_del:
        if preset_choice in load_saved_presets():
            if st.button("Delete", key="sb_del_btn"):
                delete_preset(preset_choice)
                st.rerun()

    return GlitchParams(**params_dict)


# =============================================================================
# Gallery
# =============================================================================

SAMPLE_IMAGES_DIR = Path("sample_images")


def get_sample_images():
    """Return list of sample image paths if they exist."""
    if not SAMPLE_IMAGES_DIR.exists():
        return []
    images = list(SAMPLE_IMAGES_DIR.glob("*.png")) + list(SAMPLE_IMAGES_DIR.glob("*.jpg"))
    return [str(p) for p in sorted(images)[:8]]


# =============================================================================
# Main Tabs
# =============================================================================

tabs = ["🎲 Series Generator", "🖼️ Photos → GIF", "🖼️ Gallery", "📺 GIF Overlay", "📜 CLI & Batch"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)


# -----------------------------------------------------------------------------
# Tab 1: Series Generator
# -----------------------------------------------------------------------------

with tab1:
    st.markdown('<p class="glitch-title">☠ LICHE ☠</p>', unsafe_allow_html=True)
    st.markdown("*Generate base + alt, then glitch them into a GIF with 5 random effects*")
    with st.expander("📊 Rarity levels"):
        st.markdown("""
        **By traits:** Preset from pair's rarity (highest among selected traits).
        **Random:** Rolls preset by weight — 55% common, 25% uncommon, 12% rare, 8% legendary — then picks from saved presets for that rarity.

        Suffix filenames for trait rarity: `_common`, `_uncommon`, `_rare`, `_super_rare`, `_legendary`
        """)
    st.markdown("---")

    if not SERIES_AVAILABLE:
        st.error("**Series folder not found.** Add a `series/` folder with LICHE_BASE.png, BACKGROUNDS/, EYES/, ALT_BASES/, ACCESSORIES/")
    else:
        col_ctrl, col_preview = st.columns([1, 2])
        with col_ctrl:
            acc_chance = st.slider("Accessory chance", 0.0, 1.0, 0.02, 0.01, key="sg_acc", help="2% = chance per category to get any accessory")
            eyes_chance = st.slider("Eyes chance", 0.0, 1.0, 0.1, 0.05, key="sg_eyes", help="10% = chance to get eyes")
            preset_mode = st.radio(
                "Preset",
                options=["traits", "random"],
                format_func=lambda x: "By traits (pair rarity)" if x == "traits" else "Random (55% common, 25% uncommon, 12% rare, 8% legendary)",
                key="sg_preset_mode",
            )
            series_use_mask = st.checkbox("Use auto-detect mask (protect subject)", value=False, key="sg_use_mask")
            seed = st.number_input("Random seed (optional)", min_value=0, value=0, key="sg_seed", help="0 = random each time")
            use_seed = seed if seed != 0 else None

            if st.button("⚡ GENERATE PAIR + GLITCHED GIF ⚡", use_container_width=True, key="sg_btn"):
                out_dir = Path(tempfile.gettempdir()) / "liche_series"
                out_dir.mkdir(parents=True, exist_ok=True)
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.text("Generating base + alt...")
                    progress_bar.progress(0.2)
                    img_base, img_alt, meta = generate_pair(
                        accessory_chance=acc_chance,
                        eyes_chance=eyes_chance,
                        seed=use_seed,
                    )
                    base_path = out_dir / "liche_base.png"
                    alt_path = out_dir / "liche_alt.png"
                    img_base = img_base.resize((512, 512), PILImage.Resampling.LANCZOS)
                    img_alt = img_alt.resize((512, 512), PILImage.Resampling.LANCZOS)
                    img_base.save(base_path)
                    img_alt.save(alt_path)

                    # Preset: by traits (fixed) or by RANDOM_PRESET_WEIGHTS (55/25/12/8)
                    glitch_seed = (hash(str(use_seed)) % (2**32)) if use_seed else None
                    if preset_mode == "random":
                        preset_rarity = roll_preset_rarity(seed=glitch_seed)
                        params, preset_name = get_random_preset_for_rarity(
                            preset_rarity, seed=(glitch_seed + 1) if glitch_seed is not None else None
                        )
                    else:
                        preset_rarity = meta.get("overall_rarity", "common")
                        params = get_preset_for_rarity(preset_rarity)
                        preset_name = preset_rarity

                    status_text.text(f"Applying {preset_name} preset ({preset_rarity})...")
                    progress_bar.progress(0.5)

                    status_text.text("Building GIF...")
                    gif_path = out_dir / "liche_glitched.gif"

                    def update_progress(pct, msg):
                        progress_bar.progress(0.5 + 0.5 * pct)
                        status_text.text(msg)

                    process_images_to_gif(
                        [str(base_path), str(alt_path)],
                        str(gif_path),
                        params,
                        frame_duration_ms=300,
                        static_frames_per_image=6,
                        progress_callback=update_progress,
                        bypass_mask=not series_use_mask,
                        mask_path=None,
                        max_bytes=GIF_TARGET_BYTES,
                    )

                    meta["preset_rarity"] = preset_name
                    st.session_state.series_result = str(out_dir)
                    st.session_state.series_meta = meta
                    st.session_state.series_gif = str(gif_path)
                    active = {k: v for k, v in params.__dict__.items() if isinstance(v, int) and v > 0}
                    active["chaos_level"] = round(params.chaos_level, 2)
                    if params.color_scheme != "default":
                        active["color_scheme"] = params.color_scheme
                    st.session_state.series_glitch_params = active
                    progress_bar.progress(1.0)
                    status_text.text("Done!")
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        st.markdown("---")
        st.markdown("### 📦 Batch GIF Generator")
        batch_count = st.number_input("Batch size", min_value=1, max_value=1000, value=10, key="batch_count")
        batch_out = st.text_input("Output folder", value="output/batch", key="batch_out")
        batch_seed = st.number_input("Seed (0=random)", min_value=0, value=0, key="batch_seed")
        batch_use_seed = batch_seed if batch_seed != 0 else None
        batch_use_mask = st.checkbox("Use auto-detect mask (protect subject)", value=False, key="batch_use_mask")

        if st.button("⚡ GENERATE BATCH GIFs ⚡", use_container_width=True, key="batch_btn"):
            out_dir = Path(batch_out)
            out_dir.mkdir(parents=True, exist_ok=True)
            progress = st.progress(0)
            status = st.empty()
            all_metadata = []
            try:
                for i in range(batch_count):
                    status.text(f"Generating {i + 1}/{batch_count}...")
                    progress.progress((i) / batch_count)
                    # Unique seed per item (glitch processor contaminates random state)
                    seed = (batch_use_seed or 0) + i
                    img_base, img_alt, meta = generate_pair(
                        accessory_chance=acc_chance,
                        eyes_chance=eyes_chance,
                        seed=seed,
                    )
                    img_base = img_base.resize((512, 512), PILImage.Resampling.LANCZOS)
                    img_alt = img_alt.resize((512, 512), PILImage.Resampling.LANCZOS)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t1, tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t2:
                        img_base.save(t1.name)
                        img_alt.save(t2.name)
                        base_path, alt_path = t1.name, t2.name

                    glitch_seed = (hash(str(seed)) % (2**32))
                    preset_rarity = roll_preset_rarity(seed=glitch_seed)
                    params, preset_name = get_random_preset_for_rarity(
                        preset_rarity, seed=glitch_seed + 1
                    )

                    meta["preset_rarity"] = preset_name
                    meta["seed"] = seed
                    meta["gif"] = f"liche_{i+1:04d}.gif"
                    all_metadata.append(meta)

                    gif_path = out_dir / f"liche_{i+1:04d}.gif"
                    process_images_to_gif(
                        [base_path, alt_path],
                        str(gif_path),
                        params,
                        frame_duration_ms=300,
                        static_frames_per_image=6,
                        bypass_mask=not batch_use_mask,
                        mask_path=None,
                        max_bytes=GIF_TARGET_BYTES,
                    )
                    Path(base_path).unlink(missing_ok=True)
                    Path(alt_path).unlink(missing_ok=True)

                meta_out = {}
                for m in all_metadata:
                    gif_name = m.pop("gif")
                    meta_out[gif_name] = m
                with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(meta_out, f, indent=2)

                progress.progress(1.0)
                status.text("Done!")
                st.success(f"Generated {batch_count} GIFs in `{out_dir}`")
                st.session_state.batch_result = str(out_dir)
            except Exception as e:
                st.error(f"Batch failed: {e}")
                import traceback
                st.code(traceback.format_exc())

        if "batch_result" in st.session_state:
            st.caption(f"Last batch: `{st.session_state.batch_result}`")

        if "series_result" in st.session_state and "series_gif" in st.session_state:
            res = st.session_state.series_result
            gif_path = st.session_state.series_gif
            st.markdown("### Glitched GIF")
            st.image(gif_path, use_container_width=True)
            if "series_glitch_params" in st.session_state:
                st.caption(f"Preset: {st.session_state.series_glitch_params}")
            if "series_meta" in st.session_state:
                meta = st.session_state.series_meta
                parts = [f"**{meta.get('preset_rarity', meta.get('overall_rarity', 'common')).upper()}**"]
                parts.append(f"BG: {meta.get('background', '?')} ({meta.get('background_rarity', 'common')})")
                parts.append(f"Eyes: {meta.get('eyes', '?')} ({meta.get('eyes_rarity', 'common')})")
                parts.append(f"Alt: {meta.get('alt_base', '?')}")
                if meta.get("accessories"):
                    acc = meta["accessories"]
                    rarities = meta.get("accessory_rarities", ["common"] * len(acc))
                    acc_str = ", ".join(f"{n} ({r})" for n, r in zip(acc, rarities))
                    parts.append(f"Accessories: {acc_str}")
                st.caption(" | ".join(parts))
            with open(gif_path, "rb") as f:
                st.download_button("⬇️ Download GIF", f, file_name="liche_glitched.gif", mime="image/gif", use_container_width=True, key="dl_sg_gif")
            st.success("Generated!")


# -----------------------------------------------------------------------------
# Tab 2: Photos → GIF
# -----------------------------------------------------------------------------

with tab2:
    st.markdown("## 🖼️ Photos → GIF")
    st.markdown("*Upload 2+ photos (PNG/JPG) to compile into a glitched animated GIF*")
    st.markdown("---")

    params = render_sidebar()

    photo_files = st.file_uploader(
        "Upload multiple photos (order = GIF frame order)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload 2+ images. They will be compiled into a glitched GIF in upload order.",
    )
    if photo_files and len(photo_files) >= 2:
        if "uploaded_photos" not in st.session_state:
            st.session_state.uploaded_photos = []
        st.session_state.uploaded_photos = []
        for i, f in enumerate(photo_files):
            f.seek(0)
            ext = Path(f.name).suffix or ".png"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            tmp.write(f.read())
            tmp.close()
            st.session_state.uploaded_photos.append(tmp.name)
        st.success(f"✓ Loaded {len(photo_files)} images")
        st.caption("Each image is auto-masked and gets dynamic glitch effects.")

        st.markdown("### Preview (first 3 images)")
        photo_paths = st.session_state.uploaded_photos
        prev_cols = st.columns(min(3, len(photo_paths)))
        for i, col in enumerate(prev_cols):
            if i < len(photo_paths):
                with col:
                    with open(photo_paths[i], "rb") as fp:
                        img = cv2.imdecode(np.frombuffer(fp.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Image {i+1}", use_container_width=True)

        frame_duration = st.slider(
            "Display time per image (ms)", 100, 500, 300, 50, key="gif_duration",
            help="Total time each image stays on screen",
        )
        static_frames = 6  # Fixed: 6 frames per image
        bypass_mask = st.checkbox(
            "Bypass mask (apply effects to full image)",
            value=False,
            key="gif_bypass",
            help="Unchecked = auto-detect mask from each image to protect subject",
        )
        if not bypass_mask:
            st.caption("Using auto-detect mask (edge detection) to protect subject from glitch effects.")

        st.markdown("### Generate Glitched GIF")
        if st.button("⚡ GENERATE GLITCHED GIF ⚡", use_container_width=True, key="gen_gif"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as out_tmp:
                gif_path = out_tmp.name
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(pct, msg):
                progress_bar.progress(pct)
                status_text.text(msg)

            try:
                process_images_to_gif(
                    st.session_state.uploaded_photos,
                    gif_path,
                    params,
                    frame_duration_ms=frame_duration,
                    static_frames_per_image=static_frames,
                    progress_callback=update_progress,
                    bypass_mask=bypass_mask,
                    mask_path=None,
                )
                st.session_state.processed_gif = gif_path
                st.session_state.processed_gif_as_mp4 = None
                progress_bar.progress(1.0)
                status_text.text("Done!")
                st.success("Glitched GIF ready!")
            except Exception as e:
                st.error(f"Processing failed: {e}")
                import traceback
                st.code(traceback.format_exc())

        if "processed_gif" in st.session_state:
            st.markdown("### Result")
            st.image(st.session_state.processed_gif, use_container_width=True)
            st.markdown("### Download")
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                with open(st.session_state.processed_gif, "rb") as f:
                    st.download_button(
                        "⬇️ Download as GIF",
                        f,
                        file_name="liche_glitched.gif",
                        mime="image/gif",
                        use_container_width=True,
                        key="dl_gif_gif",
                    )
            with dl_col2:
                if st.session_state.get("processed_gif_as_mp4") is None:
                    with st.spinner("Converting to MP4..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                            convert_gif_to_mp4(st.session_state.processed_gif, tmp.name)
                            st.session_state.processed_gif_as_mp4 = tmp.name
                if st.session_state.processed_gif_as_mp4:
                    with open(st.session_state.processed_gif_as_mp4, "rb") as f:
                        st.download_button(
                            "⬇️ Download as MP4",
                            f,
                            file_name="liche_glitched.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                            key="dl_gif_mp4",
                        )
    elif photo_files and len(photo_files) < 2:
        st.warning("Upload at least 2 images to create a GIF.")
    else:
        st.info("Upload 2+ photos (PNG/JPG) to compile into a glitched GIF. Order = frame order.")
        st.caption("Use the sidebar to adjust glitch effects. Zero effects = clean GIF from originals.")


# -----------------------------------------------------------------------------
# Tab 3: Gallery
# -----------------------------------------------------------------------------

with tab3:
    st.markdown("## 🖼️ Gallery")
    st.markdown("Sample images from `sample_images/` folder")
    
    sample_images = get_sample_images()
    
    if sample_images:
        n_cols = 4
        for row_start in range(0, len(sample_images), n_cols):
            cols = st.columns(n_cols)
            for c, col in enumerate(cols):
                i = row_start + c
                if i < len(sample_images):
                    with col:
                        st.image(sample_images[i], use_container_width=True, caption=Path(sample_images[i]).name)
    else:
        st.markdown("""
        **Place sample images in `sample_images/` folder:**
        - Add PNG or JPG files
        - They will appear here once added
        """)


# -----------------------------------------------------------------------------
# Tab 4: GIF Overlay
# -----------------------------------------------------------------------------

GIF_OVERLAY_DIR = Path(__file__).resolve().parent / "gif_overlay"
DEFAULT_BASE = GIF_OVERLAY_DIR / "LICHE_VOID_SQUARE.jpg"
DEFAULT_MASK = GIF_OVERLAY_DIR / "VOID_SQUARE_MASK.png"
SAVED_PRESETS_PATH = Path(__file__).resolve().parent / "saved_presets.json"


def _get_gifs_in_folder() -> list[Path]:
    """Return all GIFs in gif_overlay folder, newest first."""
    if not GIF_OVERLAY_DIR.exists():
        return []
    return sorted(GIF_OVERLAY_DIR.glob("*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)


def _get_default_gif() -> Path | None:
    """Return most recent GIF in gif_overlay folder, or None."""
    gifs = _get_gifs_in_folder()
    return gifs[0] if gifs else None


DEFAULT_GIF = _get_default_gif()  # For compatibility

with tab4:
    st.markdown("## 📺 GIF Overlay")
    st.markdown("*Overlay a GIF onto a base image using a mask. White in mask = where GIF plays.*")
    st.markdown("**Scale-to-cover:** GIF fills the mask region, preserves aspect ratio, may extend slightly beyond (mask clips).")
    st.markdown("---")

    use_defaults = st.checkbox("Use defaults from gif_overlay/", value=True, key="overlay_defaults")
    if use_defaults:
        base_file = st.file_uploader("Base image (override)", type=["png", "jpg", "jpeg"], key="overlay_base") or None
        if base_file:
            base_path = tempfile.NamedTemporaryFile(delete=False, suffix=Path(base_file.name).suffix)
            base_path.write(base_file.read())
            base_path = base_path.name
        else:
            base_path = str(DEFAULT_BASE) if DEFAULT_BASE.exists() else None

        mask_file = st.file_uploader("Mask (override)", type=["png", "jpg"], key="overlay_mask") or None
        if mask_file:
            t = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            t.write(mask_file.read())
            t.close()
            mask_path = t.name
        else:
            mask_path = str(DEFAULT_MASK) if DEFAULT_MASK.exists() else None

        gif_file = st.file_uploader("GIF (override)", type=["gif"], key="overlay_gif") or None
        if gif_file:
            t = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
            t.write(gif_file.read())
            t.close()
            gif_path = t.name
        else:
            gifs_in_folder = _get_gifs_in_folder()
            if len(gifs_in_folder) == 1:
                gif_path = str(gifs_in_folder[0])
            elif len(gifs_in_folder) > 1:
                chosen = st.selectbox(
                    "GIF to overlay",
                    options=[p.name for p in gifs_in_folder],
                    index=0,
                    key="overlay_gif_select",
                    help="Select which GIF from gif_overlay/ to use",
                )
                gif_path = str(next(p for p in gifs_in_folder if p.name == chosen))
            else:
                gif_path = None
    else:
        base_file = st.file_uploader("Base image", type=["png", "jpg", "jpeg"], key="overlay_base")
        mask_file = st.file_uploader("Mask (white = GIF region)", type=["png", "jpg"], key="overlay_mask")
        gif_file = st.file_uploader("GIF to overlay", type=["gif"], key="overlay_gif")
        base_path = None
        mask_path = None
        gif_path = None
        if base_file:
            t = tempfile.NamedTemporaryFile(delete=False, suffix=Path(base_file.name).suffix)
            t.write(base_file.read())
            t.close()
            base_path = t.name
        if mask_file:
            t = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            t.write(mask_file.read())
            t.close()
            mask_path = t.name
        if gif_file:
            t = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
            t.write(gif_file.read())
            t.close()
            gif_path = t.name

    apply_chromatic_bloom = st.checkbox(
        "Apply chromatic_bloom_uncommon to base image",
        value=True,
        key="overlay_chromatic",
        help="Add chromatic aberration + bloom to the static lich image",
    )

    if base_path and mask_path and gif_path:
        st.success("✓ All files ready")
        col1, col2 = st.columns(2)
        with col1:
            st.image(base_path, caption="Base image", use_container_width=True)
        with col2:
            st.image(mask_path, caption="Mask (white = GIF region)", use_container_width=True)

        if st.button("⚡ GENERATE OVERLAY GIF ⚡", use_container_width=True, key="overlay_btn"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as out_tmp:
                out_path = out_tmp.name
            progress = st.progress(0)
            status = st.empty()
            try:
                def update(pct, msg):
                    progress.progress(pct)
                    status.text(msg)

                overlay_gif_on_image(
                    base_path,
                    gif_path,
                    mask_path,
                    out_path,
                    progress_callback=update,
                    base_preset_name="chromatic_bloom_uncommon" if apply_chromatic_bloom else None,
                    presets_path=SAVED_PRESETS_PATH,
                )
                progress.progress(1.0)
                status.text("Done!")
                st.session_state.overlay_gif_path = out_path
                st.session_state.overlay_gif_mp4 = None
                st.success("Overlay GIF ready!")
            except Exception as e:
                st.error(f"Overlay failed: {e}")
                import traceback
                st.code(traceback.format_exc())

        if "overlay_gif_path" in st.session_state:
            st.markdown("### Result")
            st.image(st.session_state.overlay_gif_path, use_container_width=True)
            st.markdown("### Download")
            c1, c2 = st.columns(2)
            with c1:
                with open(st.session_state.overlay_gif_path, "rb") as f:
                    st.download_button("⬇️ Download GIF", f, file_name="liche_overlay.gif", mime="image/gif", use_container_width=True, key="dl_overlay_gif")
            with c2:
                if st.session_state.get("overlay_gif_mp4") is None:
                    with st.spinner("Converting to MP4..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                            convert_gif_to_mp4(st.session_state.overlay_gif_path, tmp.name)
                            st.session_state.overlay_gif_mp4 = tmp.name
                if st.session_state.overlay_gif_mp4:
                    with open(st.session_state.overlay_gif_mp4, "rb") as f:
                        st.download_button("⬇️ Download MP4", f, file_name="liche_overlay.mp4", mime="video/mp4", use_container_width=True, key="dl_overlay_mp4")
    else:
        if use_defaults:
            missing = []
            if not (base_path or DEFAULT_BASE.exists()): missing.append("base image")
            if not (mask_path or DEFAULT_MASK.exists()): missing.append("mask")
            if not (gif_path or _get_default_gif()): missing.append("GIF")
            if missing:
                st.warning(f"Missing: {', '.join(missing)}. Add files to gif_overlay/ or upload overrides.")
        else:
            st.info("Upload base image, mask, and GIF to begin.")


# -----------------------------------------------------------------------------
# Tab 5: CLI & Batch
# -----------------------------------------------------------------------------

with tab5:
    st.markdown("## 📜 CLI & Batch")
    st.markdown("*Generate pairs from the command line*")
    st.markdown("---")
    
    st.markdown("### Series Generator CLI")
    st.code("""
# Single pair → liche_base.png + liche_alt.png
python series_generator.py -o output/

# Multiple pairs (PNGs only)
python series_generator.py -n 10 -o generated/

# Batch GIFs (10 default, scale with -n)
python batch_gif.py -n 10 -o output/batch
python batch_gif.py -n 100 -o output/batch --seed 42

# Batch options
python batch_gif.py -n 50 -o out/ --preset traits --acc-chance 0.2
    """, language="bash")


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown(
    '<p class="footer-text">Liche NFT Series Generator ©</p>',
    unsafe_allow_html=True,
)
