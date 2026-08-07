# FSCV Serotonin Release Event Classifier

A local Streamlit tool for classifying raw FSCV recordings using one of three
validated, pre-trained bundles. Built for Hashemi Lab scientists who run FSCV
experiments but don't run the ML pipeline themselves.

## Setup

```bash
pip install -r requirements.txt
```

The `models/` folder ships with the app (self-contained, no OneDrive
dependency) and must stay alongside `app.py` and `fscv_core.py`:

```
fscv_app/
├── app.py
├── fscv_core.py
├── requirements.txt
├── README.md
├── assets/
│   └── imperial_logo.png
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

1. **Choose a bundle** — iPSC 3-class (MLP only), iPSC binary (RF+XGB+MLP
   ensemble), or Organoid binary (RF+XGB+MLP ensemble). Each bundle has its
   preparation config, model weights, and class count baked in — no YAML
   upload, no invalid combinations possible.
2. **Optionally rename the display labels** for that bundle's classes
   (e.g. "Event" instead of "No Event"). Cosmetic only — never changes what
   the model actually classifies.
3. **Upload one or more raw `.txt` FSCV recordings** (batch supported).
4. **Run classification.** The app windows each recording (2s windows,
   stride=5 frames, background-subtracted), classifies every window, and
   aggregates to a file-level call via the "any event override" rule: any
   non-baseline window makes the whole file non-baseline; mixed 3-class
   files go to whichever positive class has the single highest-confidence
   window.
5. **Results**: a summary table (filename, classification, confidence) for
   scanning a whole batch at a glance, plus a detail view per file — stat
   cards, the FSCV colour plot (Pablo Prieto Roca's colormap) with event
   windows highlighted, a per-window breakdown, and CSV export at both the
   batch and per-file level.

## Out of scope for v1

Event count, frequency, SNR, and baseline drift are deliberately not
implemented — they need new event-segmentation logic (grouping consecutive
positive windows into discrete events), not just reuse of existing
features. Deferred to v2 per the app spec.

## Bundles at a glance

| Bundle | Classes | Model | Held-out test F1_macro |
|---|---|---|---|
| iPSC — 3-class | Baseline / Spontaneous / Stimulated | MLP only, with confirmed 0.03 spontaneous-probability boost applied before argmax | 0.7621 |
| iPSC — Binary | Baseline / Serotonin | RF + XGB + MLP soft-voting ensemble | 0.8902 |
| Organoid — Binary | No Event / Event | RF + XGB + MLP soft-voting ensemble | 0.9044 |
