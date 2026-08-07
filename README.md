# FSCV Serotonin Release Event Classifier

A local Streamlit tool for classifying raw FSCV recordings using one of three
validated, pre-trained modes. Built for Hashemi Lab scientists who run FSCV
experiments but don't run the ML pipeline themselves.

## Setup

```bash
pip install -r requirements.txt
```

**macOS only, one-time fix:** XGBoost + PyTorch in the same process can
crash on Mac unless threading is limited. Run once:
```bash
brew install libomp
echo 'export OMP_NUM_THREADS=1' >> ~/.zprofile
echo 'export KMP_DUPLICATE_LIB_OK=TRUE' >> ~/.zprofile
source ~/.zprofile
```
Windows doesn't need this step.

Everything below ships with the app (self-contained, no OneDrive
dependency) and must stay alongside `app.py` and `fscv_core.py`:
```

fscv_app/
├── app.py
├── fscv_core.py
├── requirements.txt
├── README.md
├── .streamlit/
│ └── config.toml
├── assets/
│ └── imperial_logo.png
├── data/
│ └── voltage_values.csv
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
(Windows: `py -m streamlit run app.py`. macOS: `python3 -m streamlit run app.py`.)

Opens at `localhost:8501`.

## What it does

1. **Choose a mode** — Serotonergic Spheroid 3-Class (MLP only), Serotonergic
   Spheroid Binary (RF+XGB+MLP ensemble), or Gut Organoid Binary (RF+XGB+MLP
   ensemble). Each mode has its preparation config, model weights, and class
   count baked in — no YAML upload, no invalid combinations possible.
2. **Optionally rename the display labels** for that mode's classes.
   Cosmetic only — never changes what the model actually classifies.
3. **Upload one or more `.txt` FSCV recordings** (batch supported).
4. **Run classification.** The app windows each recording (2s windows,
   stride=5 frames, background-subtracted), classifies every window, and
   aggregates to a file-level call via the "any event override" rule: any
   non-baseline window makes the whole file non-baseline; mixed 3-class
   files go to whichever positive class has the single highest-confidence
   window.
5. **Results**: a summary table (filename, classification, confidence) for
   scanning a whole batch at a glance, plus a detail view per file — stat
   cards (each with a hover-info icon explaining that metric), the FSCV
   colour plot (Pablo Prieto Roca's colormap, real voltage values on the
   y-axis) with event windows highlighted, a per-window breakdown, and CSV
   export (simple summary or full stats) at both the batch and per-file level.

## Known display correction — sign convention

Bettina identified that the raw `.txt` recordings use a sign convention
that renders backwards on Pablo's colormap (oxidation shows blue, reduction
shows green — the reverse of the expected polarity). Since every file used
to label, window, and train the currently deployed models has been
**consistently** non-inverted throughout, model performance and the
reported F1 scores are unaffected — this is a display issue, not a
classification issue.

## Out of scope for v1

Event count, frequency, SNR, and baseline drift are deliberately not
implemented — they need new event-segmentation logic (grouping consecutive
positive windows into discrete events), not just reuse of existing
features. Deferred to v2 per the app spec.

## Modes at a glance

| Mode | Classes | Model | Held-out test F1_macro |
|---|---|---|---|
| Serotonergic Spheroid — 3-Class | Baseline / Spontaneous / Stimulated | MLP only, with confirmed 0.03 spontaneous-probability boost applied before argmax | 0.7621 |
| Serotonergic Spheroid — Binary | Baseline / Serotonin | RF + XGB + MLP soft-voting ensemble | 0.8902 |
| Gut Organoid — Binary | Baseline / Serotonin (internally No Event / Event) | RF + XGB + MLP soft-voting ensemble | 0.9044 |
