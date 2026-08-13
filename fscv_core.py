"""
fscv_core.py — shared logic for the FSCV Event Classifier app.

Contains:
  - Bundle definitions (config + model paths), mirroring fscv_config_ipsc.yaml
    and fscv_config_organoid.yaml exactly, baked in rather than uploaded.
  - Windowing logic for a freshly uploaded recording (adapted from
    make_windows_ipsc_binary.py / make_windows_organoid.py — same math,
    stripped of the training-only scaffolding: no labels CSV, no folder
    scanning, just slide a window across whatever file was uploaded).
  - The 17-feature extract() function, identical to extract_features_ipsc_binary.py
    / extract_features_organoid_17.py.
  - Pablo Prieto Roca's custom FSCV colormap (NeuroStemVolt, 2026), ported
    directly from the labelling app.
  - Model loading (cached) and per-window / per-file prediction, including
    the soft-voting ensemble for both binary bundles and the confirmed
    spontaneous/stimulated probability boost for the iPSC 3-class MLP.

Sign convention: load_recording() inverts the raw array at the source
(matching the -arr fix applied in make_windows_*.py and the labelling apps).
All three bundles' models were retrained on sign-corrected data, so this is
now safe to do at the source rather than as a display-only workaround in
app.py -- the old arr_display = -arr / extract(-windows[...]) patches in
app.py should be removed now that this is fixed here.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.colors as mcolors
from scipy.stats import skew, kurtosis

# ============================================================
# Constants shared across all bundles
# ============================================================

N_VOLTAGE_PTS = 1100
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ============================================================
# Voltage lookup table — maps each of the 1100 row indices to its
# real voltage value (Jackson waveform: 0.2V -> 1.0V -> -0.1V -> 0.2V).
# Non-monotonic (rises, falls, rises again), so this can't be shown as a
# simple linear axis relabel — used to build custom tick labels on the
# colour plot instead, at the actual row position each voltage occurs.
# ============================================================
VOLTAGE_VALUES = np.loadtxt(
    os.path.join(os.path.dirname(__file__), "data", "voltage_values.csv"),
    delimiter=",",
)

# ============================================================
# Preparation configs — baked in from fscv_config_ipsc.yaml /
# fscv_config_organoid.yaml. No YAML upload in the app; these
# are the exact values confirmed in the project.
# ============================================================

IPSC_CFG = {
    "fscv_hz": 10.0,
    "stride": 5,
    "max_nothing": 50,
    "v_oxidation_start": 200,
    "v_oxidation_end": 400,
    "v_reduction_start": 800,
    "v_reduction_end": 1000,
    "balance_ratio": 2,
}

ORGANOID_CFG = {
    "fscv_hz": 10.0,
    "stride": 5,
    "max_nothing": 50,
    "v_oxidation_start": 150,
    "v_oxidation_end": 300,
    "v_reduction_start": 800,
    "v_reduction_end": 1000,
    "balance_ratio": 2,
}

# ============================================================
# The three validated bundles. All three retrained post sign-fix,
# post relabelling (organoid) -- see project logbook for full trajectory.
# ============================================================

BUNDLES = {
    "ipsc_3class": {
        "key": "ipsc_3class",
        "menu_label": "Serotonergic Spheroid — 3-Class",
        "config": IPSC_CFG,
        "class_names": ["Baseline", "Spontaneous", "Stimulated"],
        "default_display_names": ["Baseline", "Spontaneous", "Stimulated"],
        "n_classes": 3,
        "model_type": "mlp_only",
        "mlp_path": os.path.join(MODELS_DIR, "mlp_model_ipsc_3class.pkl"),
        # Two-part boost, tuned via sweep_boost_oof.py against CV out-of-fold
        # predictions (not the test set), then confirmed once against the
        # held-out test set. Replaces the old single-boost 0.03 value used
        # pre-sign-fix.
        "boost": [
            {"class_idx": 1, "factor": 0.10},  # spontaneous
            {"class_idx": 2, "factor": 1.00},  # stimulated (no-op, kept explicit)
        ],
        "test_f1_macro": 0.7267,
    },
    "ipsc_binary": {
        "key": "ipsc_binary",
        "menu_label": "Serotonergic Spheroid — Binary",
        "config": IPSC_CFG,
        "class_names": ["Baseline", "Serotonin"],
        "default_display_names": ["Baseline", "Serotonin"],
        "n_classes": 2,
        "model_type": "ensemble",
        "mlp_path": os.path.join(MODELS_DIR, "mlp_model_ipsc_binary.pkl"),
        "rf_path": os.path.join(MODELS_DIR, "rf_model_ipsc_binary.pkl"),
        "xgb_path": os.path.join(MODELS_DIR, "xgb_model_ipsc_binary.pkl"),
        "test_f1_macro": 0.8712,
    },
    "organoid_binary": {
        "key": "organoid_binary",
        "menu_label": "Organoid — Binary",
        "config": ORGANOID_CFG,
        "class_names": ["No Event", "Event"],
        "default_display_names": ["Baseline", "Serotonin"],
        "n_classes": 2,
        "model_type": "ensemble",
        "mlp_path": os.path.join(MODELS_DIR, "mlp_model_organoid.pkl"),
        "rf_path": os.path.join(MODELS_DIR, "rf_model_organoid.pkl"),
        "xgb_path": os.path.join(MODELS_DIR, "xgb_model_organoid.pkl"),
        "test_f1_macro": 0.8650,
    },
}

FEATURE_ORDER = [
    "peak_current", "peak_voltage", "peak_width", "trough_current",
    "auc_sero", "auc_full", "mean", "std", "skewness", "kurtosis",
    "time_change", "time_of_max", "rise_time", "decay_time",
    "ox_red_ratio", "rise_slope", "ox_red_lag",
]

FEATURE_DISPLAY_NAMES = {
    "peak_current": "Amplitude",
    "rise_time": "Rise time",
    "decay_time": "Decay time",
    "peak_width": "Peak width (FWHM)",
    "auc_sero": "AUC (oxidation)",
    "auc_full": "AUC (full)",
    "ox_red_ratio": "Ox/Red ratio",
    "ox_red_lag": "Ox/Red lag",
}


# ============================================================
# Windowing (inference-time, no labels — adapted from
# make_windows_ipsc_binary.py / make_windows_organoid.py)
# ============================================================

def load_recording(file_obj):
    """Load an uploaded .txt FSCV recording into a 2D array (voltage x time).
    Inverted per Bettina's guidance: the raw .txt files use a sign
    convention that renders backwards on Pablo's colormap. This flips the
    sign at the source, matching the fix applied in make_windows_*.py and
    the labelling apps -- all three deployed bundles were retrained on
    sign-corrected data, so this is safe now (previously this was handled
    as a display-only workaround in app.py; that workaround should be
    removed now that this is fixed here, to avoid inverting twice)."""
    arr = np.loadtxt(file_obj)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return -arr


def window_recording(arr, cfg, bg_subtract=True):
    """
    Slide a window across the whole uploaded recording.
    Same window_frames/stride math as training, no label-driven looping
    (there's nothing to look up — this is inference on a fresh file).
    Returns: list of window arrays, list of (start_frame, end_frame) tuples.
    """
    fscv_hz = cfg["fscv_hz"]
    stride = cfg["stride"]
    window_frames = int(2.0 * fscv_hz)
    bg_frames = int(5.0 * fscv_hz)

    if bg_subtract and arr.shape[1] > bg_frames:
        arr = arr - arr[:, :bg_frames].mean(axis=1, keepdims=True)

    nT = arr.shape[1]
    windows, spans = [], []
    for f0 in range(0, max(0, nT - window_frames + 1), stride):
        w = arr[:, f0:f0 + window_frames]
        if w.shape[1] != window_frames:
            continue
        windows.append(w)
        spans.append((f0, f0 + window_frames))
    return windows, spans, window_frames, arr  # arr returned post-bg-subtraction, for plotting


# ============================================================
# Feature extraction — identical to extract_features_ipsc_binary.py /
# extract_features_organoid_17.py
# ============================================================

def extract(arr, cfg):
    """Extract 17 features from a window array using config voltage indices."""
    v0 = cfg["v_oxidation_start"]
    v1 = cfg["v_oxidation_end"]
    r0 = cfg["v_reduction_start"]
    r1 = cfg["v_reduction_end"]

    ox = arr[v0:v1, :]
    red = arr[r0:r1, :]
    flat = arr.flatten()

    ox_trace = ox.mean(axis=0)
    red_trace = red.mean(axis=0)
    n_frames = len(ox_trace)

    peak_frame = ox_trace.argmax()
    peak_val = ox_trace[peak_frame]
    red_frame = red_trace.argmin()

    rise_time = peak_frame

    frames_left = n_frames - 1 - peak_frame
    decay_time = (peak_val - ox_trace[-1]) / frames_left if frames_left > 0 else 0.0

    rise_slope = (peak_val - ox_trace[0]) / max(peak_frame, 1)
    ox_red_lag = red_frame - peak_frame

    raw_ratio = ox.max() / (abs(red.min()) + 1e-3)
    ox_red_ratio = float(np.clip(raw_ratio, -10, 10))

    return {
        "peak_current": float(ox.max()),
        "peak_voltage": float(ox.mean(axis=1).argmax() + v0),
        "peak_width": float((ox_trace > ox_trace.max() * 0.5).sum()),
        "trough_current": float(red.min()),
        "auc_sero": float(np.abs(ox).sum()),
        "auc_full": float(np.abs(arr).sum()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "skewness": float(skew(flat)),
        "kurtosis": float(kurtosis(flat)),
        "time_change": float(arr.mean(axis=0)[-5:].mean() - arr.mean(axis=0)[:5].mean()),
        "time_of_max": float(arr.mean(axis=0).argmax()),
        "rise_time": float(rise_time),
        "decay_time": float(decay_time),
        "ox_red_ratio": float(ox_red_ratio),
        "rise_slope": float(rise_slope),
        "ox_red_lag": float(ox_red_lag),
    }


# ============================================================
# Pablo Prieto Roca's custom FSCV colormap (NeuroStemVolt, 2026)
# Ported directly from the labelling app's PLOT_SETTINGS class.
# ============================================================

class PlotSettings:
    """Custom colourmap settings for FSCV data visualisation."""

    def __init__(self):
        self.custom = self._get_continuous_cmap(
            ['#001524', '#002f5e', '#f4c300', '#a84900',
             '#64005f', '#21AE62', '#00751c', '#00ff00'],
            [0, 0.2478, 0.3805, 0.6555, 0.701, 0.7603, 0.7779, 1]
        )

    def get_norm(self, data, clim=None, vmax=None):
        if vmax is not None:
            return mcolors.Normalize(vmin=-(2 / 3) * vmax, vmax=vmax)
        if clim is None:
            clim = np.nanmax(np.abs(data))
        return mcolors.Normalize(vmin=-(2 / 3) * clim, vmax=clim)

    def _get_continuous_cmap(self, hex_list, float_list=None):
        rgb_list = [self._rgb_to_dec(self._hex_to_rgb(i)) for i in hex_list]
        if float_list is None:
            float_list = list(np.linspace(0, 1, len(rgb_list)))
        cdict = dict()
        for num, col in enumerate(['red', 'green', 'blue']):
            col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]]
                        for i in range(len(float_list))]
            cdict[col] = col_list
        return mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)

    def _hex_to_rgb(self, value):
        value = value.strip("#")
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def _rgb_to_dec(self, value):
        return [v / 256 for v in value]


PLOT_SETTINGS = PlotSettings()


# ============================================================
# MLP architecture — must match train_models_*.py exactly
# ============================================================

class MLP(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_VOLTAGE_PTS * 20, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Model loading (cache-friendly — call via st.cache_resource
# from the app so this only runs once per session)
# ============================================================

def load_bundle_models(bundle):
    """Load whichever model(s) a bundle needs. Returns a dict of loaded objects."""
    import pickle

    loaded = {}

    mlp_data = pickle.load(open(bundle["mlp_path"], "rb"))
    mlp_model = MLP(bundle["n_classes"])
    mlp_model.load_state_dict(mlp_data["model_state"])
    mlp_model.eval()
    loaded["mlp"] = {"model": mlp_model, "mean": mlp_data["mean"], "std": mlp_data["std"]}

    if bundle["model_type"] == "ensemble":
        rf_data = pickle.load(open(bundle["rf_path"], "rb"))
        xgb_data = pickle.load(open(bundle["xgb_path"], "rb"))
        loaded["rf"] = rf_data["model"]
        loaded["xgb"] = xgb_data["model"]

    return loaded


# ============================================================
# Per-file prediction pipeline
# ============================================================

def predict_windows(bundle, models, windows):
    """
    Run the bundle's model(s) on every window of one uploaded file.
    Returns: proba array (n_windows, n_classes).
    """
    X_raw = np.array([w.flatten() for w in windows], dtype=np.float32)

    mlp_info = models["mlp"]
    X_norm = (X_raw - mlp_info["mean"]) / (mlp_info["std"] + 1e-8)
    with torch.no_grad():
        logits = mlp_info["model"](torch.FloatTensor(X_norm))
        mlp_proba = torch.softmax(logits, dim=1).numpy()

    if bundle["model_type"] == "mlp_only":
        proba = mlp_proba.copy()
        boosts = bundle.get("boost")
        if boosts:
            for b in boosts:
                proba[:, b["class_idx"]] *= b["factor"]
            proba = proba / proba.sum(axis=1, keepdims=True)
        return proba

    # Ensemble bundles: need 17 engineered features for RF/XGB
    cfg = bundle["config"]
    feat_dicts = [extract(w, cfg) for w in windows]
    X_feat = np.array([[d[k] for k in FEATURE_ORDER] for d in feat_dicts], dtype=np.float32)

    rf_proba = models["rf"].predict_proba(X_feat)
    xgb_proba = models["xgb"].predict_proba(X_feat)
    ensemble_proba = (rf_proba + xgb_proba + mlp_proba) / 3.0
    return ensemble_proba


def aggregate_file(window_preds, window_proba, n_classes):
    """
    'Any event override': any non-baseline window -> whole file gets that
    class. Mixed files (3-class only) go to whichever positive class has
    the single highest confidence among its own positive windows.
    Confidence placeholder = single winning-class probability (not yet
    confirmed by Dr Hashemi — swap here once confirmed).
    Returns: file_class_idx, confidence, representative_window_idx (or
    None if baseline), positive_window_indices (for colour-plot highlighting).
    """
    positive_mask = window_preds != 0
    positive_idx = np.where(positive_mask)[0]

    if len(positive_idx) == 0:
        # Baseline: confidence placeholder = mean baseline-class probability
        conf = float(window_proba[:, 0].mean()) if len(window_proba) else 0.0
        return 0, conf, None, positive_idx

    best_class, best_conf, best_window = None, -1.0, None
    for c in range(1, n_classes):
        idxs = positive_idx[window_preds[positive_idx] == c]
        if len(idxs) == 0:
            continue
        confs = window_proba[idxs, c]
        local_best = idxs[np.argmax(confs)]
        if window_proba[local_best, c] > best_conf:
            best_class, best_conf, best_window = c, float(window_proba[local_best, c]), int(local_best)

    return best_class, best_conf, best_window, positive_idx