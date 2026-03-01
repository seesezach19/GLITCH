# Liche

NFT series generator — random base + alt compositions from layered assets.

**Each run produces 2 images:** one with the base character + attributes, one with an alt base + the same attributes (background, eyes, accessories).

---

## Features

- **Layered composition**: Background → Base/Alt → Eyes → Accessories
- **Random generation**: Picks from series/ folder (backgrounds, eyes, hats, sunglasses, etc.)
- **CLI + Streamlit**: Generate from command line or web UI
- **Batch mode**: Generate multiple pairs with one command

---

## Installation

```bash
cd NFT
pip install -r requirements.txt
```

---

## Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Series Folder Structure

Place your assets in a `series/` folder:

```
series/
├── LICHE_BASE.png          # Base character
├── BACKGROUNDS/            # PNG backgrounds
├── EYES/                   # Eye variants
├── ALT_BASES/              # Alternate bases (CYAN/, GREEN/, MAGENTA/)
│   ├── CYAN/
│   ├── GREEN/
│   └── MAGENTA/
└── ACCESSORIES/
    ├── CIGARETTE/
    ├── HATS/
    ├── SUNGLASSES/
    └── TEETH/
```

---

## CLI Usage

```bash
# Single pair → liche_base.png + liche_alt.png
python series_generator.py -o output/

# Multiple pairs
python series_generator.py -n 10 -o generated/

# With options
python series_generator.py -n 5 -o out/ --acc-chance 0.7 --seed 42
```

---

## Project Structure

```
NFT/
├── app.py                 # Streamlit app
├── series_generator.py    # Layer composition & random generation
├── requirements.txt
├── series/                # Layered assets (see structure above)
├── sample_images/         # Optional gallery images
└── .streamlit/            # Streamlit config
```

---

## Tech Stack

- **Streamlit** — UI
- **Pillow** — image composition

---

## License

MIT.
