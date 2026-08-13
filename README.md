# FSCV Event Classifier

A local Streamlit tool for classifying FSCV recordings using one of three
validated, pre-trained modes. Built for Hashemi Lab scientists who run FSCV
experiments but don't run the ML pipeline themselves.

## Setup

```bash
pip install -r requirements.txt
```

The `models/` folder ships with the app (self-contained, no OneDrive
dependency) and must stay alongside `app.py` and `fscv_core.py`, along with
`assets/`, `data/`, and `.streamlit/`:

```
fscv_app/
├── app.py
├── fscv_core.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml            # theme (Imperial pink accent)
├── assets/
│   └── imperial_logo.png
├── data/
│   └── voltage_values.csv     # 1100-point Jackson waveform voltage lookup
└── models/
    ├── mlp_model_ipsc_3class.pkl
    ├── mlp_model_ipsc_binary.pkl
    ├── mlp_model_organoid.pkl
    ├── rf_model_ipsc_binary.pkl
    ├── rf_model_organoid.pkl
    ├── xgb_model_ipsc_binary.pkl
    └── xgb_model_organoid.pkl
```

## Run

```bash
streamlit run app.py
```

Opens at `localhost:8501`.

## What it does

1. **Choose a mode** — Serotonergic Spheroid 3-class (MLP only),
   Serotonergic Spheroid binary (RF+XGB+MLP ensemble), or Organoid binary
   (RF+XGB+MLP ensemble). Each mode has its preparation config, model
   weights, and class count baked in — no YAML upload, no invalid
   combinations possible.
2. **Optionally rename the display labels** for that mode's classes
   (e.g. "Event" instead of "No Event"). Cosmetic only — never changes what
   the model actually classifies.
3. **Upload one or more `.txt` FSCV recordings** (batch supported). Files
   should already be passed through a Butterworth low-pass filter (5000 Hz
   cutoff, five times the 1000 V/s scan rate) before uploading — the app
   applies background subtraction automatically, but not this filtering.
4. **Run classification.** The app windows each recording (2s windows,
   stride=5 frames), classifies every window, and aggregates to a
   file-level call via the "any event override" rule: any non-baseline
   window makes the whole file non-baseline; mixed 3-class files go to
   whichever positive class has the single highest-confidence window.
5. **Results**: a confidence-threshold slider (default 0.70, live
   adjustable) splits the batch into three tabs — high confidence, below
   threshold, and all results — each with its own CSV export, plus a
   combined full-stats CSV across the whole batch. A detail view per file
   shows stat cards (with units), the FSCV colour plot (Pablo Prieto
   Roca's colormap) with event windows highlighted, a starred peak-
   amplitude marker with a horizontal guide line and highlighted voltage
   label showing exactly where it crosses the y-axis, a per-window
   breakdown, and per-file CSV export.

## Bundles at a glance

| Mode | Classes | Model | Held-out test F1_macro |
|---|---|---|---|
| Serotonergic Spheroid — 3-class | Baseline / Spontaneous / Stimulated | MLP only, two-part probability boost (spontaneous ×0.10, stimulated ×1.00) tuned via CV out-of-fold sweep, applied before argmax | 0.7267 (MCC 0.664) |
| Serotonergic Spheroid — Binary | Baseline / Serotonin | RF + XGB + MLP soft-voting ensemble | 0.8712 |
| Organoid — Binary | No Event / Event | RF + XGB + MLP soft-voting ensemble | 0.8650 |


