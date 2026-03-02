# Liche NFT Project — Full Overview

**Purpose:** Generate glitched animated NFT art from layered compositions. Skull/lich character with backgrounds, eyes, accessories. Output: GIFs + metadata for NFT minting.

---

## 1. Project Structure

```
NFT/
├── app.py                 # Streamlit UI (main entry: python -m streamlit run app.py)
├── batch_gif.py          # CLI batch generator (python batch_gif.py -n 100 -o output/batch)
├── series_generator.py   # Layer composition (background + base + eyes + accessories)
├── glitch_processor.py   # Core: 40+ CPU effects + GlitchParams (100+ fields)
├── shader_processor.py   # OpenGL/GLSL shader chain (moderngl), auto-discovers shader_*_intensity
├── gif_processor.py      # Images → GIF pipeline, transitions, WebP/APNG export
├── transition_effects.py  # 12 A↔B boundary transitions (band wipe, slit-scan, etc.)
├── gif_overlay_utils.py   # Overlay GIF onto base image using mask (e.g. lich holding screen)
├── glitch_presets.py     # Rarity weights (55% common, 25% uncommon, etc.)
├── glitch_preset_config.py # Boundary-aware preset config (for process_frame_boundary_aware)
├── saved_presets.json    # User presets: effect intensities per preset name
├── showcase_*.py         # MP4 showcases (backgrounds, effects, accessories, eyes)
├── SERIES/               # Layer assets
│   ├── LICHE_BASE.png
│   ├── BACKGROUNDS/      # PNGs, rarity via filename suffix _common, _rare, etc.
│   ├── EYES/
│   ├── ALT_BASES/        # Color variants (CYAN/, RED/, etc.)
│   └── ACCESSORIES/      # CIGARETTE/, HATS/, SUNGLASSES/, etc.
├── SHADERS/              # GLSL fragment shaders (.frag)
│   ├── kaleido_grid.frag, stripe_shift.frag, block_smear.frag, palette_rainbow.frag
│   ├── dot_matrix.frag, pixel_rain.frag, liquid_metal.frag, data_corruption.frag
│   ├── vhs_rewind.frag, holographic_foil.frag, crystalline_frost.frag, void_tendrils.frag
│   └── ... (22 total)
├── gif_overlay/          # Overlay feature assets
│   ├── LICHE_VOID_SQUARE.jpg   # Base image (lich holding black square)
│   ├── VOID_SQUARE_MASK.png    # White = where GIF plays
│   └── *.gif                    # GIFs to overlay
└── output/
    ├── batch/            # static/, animated/, metadata/ per token
    └── *.mp4             # Showcase videos
```

---

## 2. Core Data Flow

### 2.1 Layer Composition (series_generator.py)

```
generate_pair(accessory_chance, eyes_chance, seed)
  → _collect_layer_paths()  # Discover SERIES/ folders
  → _weighted_choice()      # Pick layers by rarity (common=19, uncommon=10, rare=5, etc.)
  → _composite_layers()     # Alpha composite: bg + behind_base + base + eyes + accessories
  → Returns (img_base, img_alt, metadata)
```

- **Base:** LICHE_BASE.png
- **Alt:** Random from ALT_BASES/* (e.g. LICHE_CYAN, LICHE_RED_BLACK)
- **Background:** Weighted by filename suffix (_common, _rare, _legendary)
- **Eyes:** eyes_chance (default 10%), else transparent
- **Accessories:** Per-category chance (default 2%), order: cigarette, cracks, hats, sunglasses, teeth

### 2.2 GIF Pipeline (gif_processor.py)

```
process_images_to_gif(image_paths, output_path, params, ...)
  → _load_images()         # Resize to match first image
  → For each image i, for each frame k in static_frames_per_image:
       mask = mask_path OR auto-detect OR zeros (bypass_mask)
       fp = vary_params_for_frame(params, idx)  # Per-frame intensity scaling
       if is_boundary (first frame of new image): apply transition (band_wipe, etc.)
       out = process_frame(frame, mask, fp, idx, prev)
  → _save_gif_under_size() # Global palette, smarter frame decimation, color reduction
  → Optional: save_as_webp(), save_as_apng() for higher-quality animated export
```

- **Default:** 2 images (base + alt), 6 frames each = 12-frame GIF, 300ms per image
- **Boundary:** When switching from image A to B, one of 12 transition effects can run
- **Mask:** 255 = protected (subject), 0 = glitch zone. `mask_path` = custom upload; else auto-detect (edge detection + background subtraction)
- **Output:** RESOLUTION_OPTIONS (256–1024px), Floyd-Steinberg dithering option, WebP/APNG download

### 2.3 Glitch Processing (glitch_processor.py + shader_processor.py)

```
process_frame(frame, mask, params, frame_idx, prev_frame)
  → make_soft_mask(mask, mask_soft_edge)
  → Apply CPU effects in order (only where mask allows):
       - Geometry: frame_drift, melting, block_tear, flow_warp, voronoi_shatter, displacement_map
       - Color: rgb_shift, chromatic_aberration, bitcrush, chroma_dropout, color_halftone
       - Overlays: scanlines, film_grain, tv_static, ordered_dither
       - Temporal: ghost_trail, temporal_echo, echo_crown, frame_drop
       - Foreground: subject_invert, subject_particles, edge_pulse
  → shader_processor.apply_shaders_chain() — GLSL effects (kaleido_grid, stripe_shift, dot_matrix, etc.)
  → Composite: mask_soft * subject + (1-mask_soft) * background
```

- **GlitchParams:** Dataclass with 100+ fields. Each effect has intensity 0–10.
- **vary_params_for_frame:** Scales intensities by event cycle (subtle → peak at frame 6 → decay)
- **prev_frame:** Used by ghost_trail, slit_scan, echo_crown, temporal_echo, frame_drop (temporal effects)
- **GLSL shaders:** Auto-discovered from `shader_*_intensity` fields; loaded from SHADERS/*.frag

---

## 3. Key Code Patterns

### GlitchParams (glitch_processor.py)

```python
@dataclass
class GlitchParams:
    rgb_shift_intensity: int = 0
    chromatic_aberration: int = 0
    # ... 100+ effect params, all int 0-10 except:
    chaos_level: float = 0.0
    color_scheme: str = "default"
    mask_sensitivity: float = 0.0
    mask_soft_edge: int = 0
    mask_opacity: float = 1.0
    # shader_*_intensity: kaleido_grid, stripe_shift, block_smear, palette_rainbow, dot_matrix, etc.
    # Transition effects: transition_band_wipe, transition_diagonal_rip, etc.
```

### Preset Loading

```python
# From saved_presets.json — preset name must end with _common, _uncommon, _rare, _legendary
saved = json.load("saved_presets.json")
params = GlitchParams(**{k: v for k, v in saved[name].items() if k in GlitchParams.__dataclass_fields__})
```

### Mask Convention

- **255** = protected (subject, no glitch)
- **0** = glitch zone (effects apply)
- `bypass_mask=True` → mask = zeros (effects everywhere)
- `mask_path` = custom uploaded mask (GIF maker + overlay tabs)
- `create_auto_mask()` → edge detection, returns binary mask (used when no mask_path)

---

## 4. Effect Categories

### CPU Effects (glitch_processor.py)

| Category | Effects |
|----------|---------|
| **Core** | rgb_shift, chromatic_aberration, scanlines, noise, pixelation, datamosh, melting |
| **Spatial** | flow_warp, slit_scan, parallax_split, voronoi_shatter, displacement_map |
| **Temporal** | echo_crown, strobe_phase, frame_drop, ghost_trail, temporal_echo |
| **Palette** | palette_cycle, chroma_collapse, ordered_dither, color_halftone |
| **Occult** | sigil_ring, phylactery_glow, abyss_window |
| **Mathy** | fft_jitter, moire_grid |
| **Transitions** | 12 types (band_wipe, diagonal_rip, slit_scan_swap, etc.) |

### GLSL Shaders (SHADERS/*.frag)

| shader_*_intensity | Description |
|--------------------|-------------|
| kaleido_grid | Mirror-tiled portal/panel kaleidoscope layout |
| stripe_shift | Horizontal band displacement with per-band X shift |
| block_smear | Codec-broken block swap/smear geometry |
| palette_rainbow | Posterize + luminance-mapped rainbow banding |
| dot_matrix | LED panel / crisp pixel grid (luminance-driven pulse) |
| pixel_rain, liquid_metal, data_corruption, vhs_rewind, holographic_foil | |
| crystalline_frost, void_tendrils, spectral_prism, soul_fire, electric_arc | |
| dimensional_rift, glitch_hologram, caustic_flow, thermal_distort, hexagonal_warp, necrotic_iridescent_flow | |

---

## 5. GIF Overlay Feature (gif_overlay_utils.py)

```
overlay_gif_on_image(base_path, gif_path, mask_path, output_path, base_preset_name, presets_path)
  → Load base image
  → Optionally apply preset (e.g. chromatic_bloom_uncommon) to base; ordered_dither forced to 1
  → Get mask bbox (white = GIF region)
  → Scale GIF to cover bbox (preserve AR, may overflow; mask clips)
  → For each GIF frame: composite onto base using mask alpha
  → Save as GIF with original frame durations
```

- **Scale-to-cover:** `scale = max(bbox_w/gif_w, bbox_h/gif_h)` — fills mask, preserves aspect ratio
- **Streamlit tab:** Upload or use defaults from gif_overlay/, select GIF from dropdown if multiple, checkbox for chromatic bloom

---

## 6. Batch Output (batch_gif.py)

```
python batch_gif.py -n 100 -o output/batch [--use-mask] [--preset random|traits]
```

- **Output:** `output/batch/static/0001.png`, `animated/0001.gif`, `metadata/0001.json`
- **Metadata:** name, traits (Background, Eyes, Accessories, etc.), image URL, animation_url, seed
- **Preset:** `random` = roll rarity (55/25/12/8), pick matching preset from saved_presets.json

---

## 7. Streamlit App (app.py)

**Tabs:**
1. **Series Generator** — Generate pair + glitched GIF, batch generator
2. **Custom GIF** — Upload 2+ images, optional custom mask (white=protected), apply glitch effects, sidebar has 100+ sliders. Output: GIF + WebP + APNG download.
3. **Gallery** — Sample images from sample_images/
4. **GIF Overlay** — Base + mask + GIF → composite with optional chromatic bloom
5. **CLI & Batch** — Command reference

**Custom GIF:** Resolution selector (256–1024px), Floyd-Steinberg dithering, bypass mask checkbox. Mask upload overrides auto-detect when provided.
**Preset flow:** Select preset from saved_presets.json → sliders override with preset values → save/delete presets

---

## 8. Dependencies

- **OpenCV** (cv2) — image load, resize, remap, Canny, etc.
- **NumPy** — arrays, FFT, random
- **Pillow** (PIL) — GIF/WebP/APNG save, quantize, alpha composite
- **moderngl** — OpenGL 3.3 context for GLSL shaders
- **Streamlit** — UI
- **moviepy** — GIF → MP4 (for download)

---

## 9. Rarity System

- **Filename suffix:** `_common`, `_uncommon`, `_rare`, `_super_rare`, `_legendary`
- **rarity_config.json:** Override per file path
- **Weights:** common=19, uncommon=10, rare=5, super_rare=2, legendary=1
- **Preset roll:** 55% common, 25% uncommon, 12% rare, 8% legendary → pick preset from saved_presets.json with matching suffix

---

## 10. Notable Implementation Details

- **Event cycle:** EVENT_CYCLE=12, EVENT_PEAK_FRAME=6 — effects peak at frame 6, decay after
- **Global palette:** Random sampling + median cut (method=2) for perceptually pleasing GIF palettes
- **GIF size:** Target 700KB; smarter frame decimation (drop similar consecutive frames first), then color reduction (256→128→64→32)
- **Transition pick:** When multiple transition params > 0, `pick_transition()` chooses by weighted random
- **Auto-mask:** `create_auto_mask()` — background subtraction + edge detection, face_safe option
- **Shader auto-discovery:** `shader_processor` scans GlitchParams for `shader_*_intensity`, loads matching SHADERS/*.frag
- **Layer cache:** `_collect_layer_paths()` caches SERIES/ scan to avoid redundant filesystem reads
