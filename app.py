"""
FSCV Event Classifier — Streamlit app.

Three fixed, validated modes (iPSC 3-class, iPSC binary, organoid binary).
No YAML upload, no free prep x mode combination — pick a mode, optionally
rename the display labels, upload recordings, get classifications, colour
plots, stat cards, and CSV export.

Run: streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from fscv_core import (
    BUNDLES, PLOT_SETTINGS, VOLTAGE_VALUES,
    load_recording, window_recording, extract,
    load_bundle_models, predict_windows, aggregate_file,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(APP_DIR, "assets", "imperial_logo.png")

st.set_page_config(page_title="FSCV Event Classifier", layout="wide", page_icon="🧠")

# ------------------------------------------------------------------
# Styling — Imperial pink accent (#C2185B), matching the presentation
# deck's established palette, laid over a clean lab-tool aesthetic.
# ------------------------------------------------------------------
ACCENT = "#c71585"
ACCENT_BAR = "#FF69B4"

st.markdown(f"""
<style>
.stat-card {{
    background: #ffffff; border: 1px solid #ececec; border-radius: 12px;
    padding: 16px 14px; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.stat-label {{ font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }}
.stat-value {{ font-size: 24px; font-weight: 700; color: #1a1a1a; margin-top: 4px; }}
.badge {{
    display: inline-block; padding: 5px 16px; border-radius: 999px;
    font-weight: 600; font-size: 13px; color: white; letter-spacing: 0.02em;
}}
.badge-baseline {{ background: #8a8f98; }}
.badge-c1 {{ background: {ACCENT}; }}
.badge-c2 {{ background: {ACCENT_BAR}; }}
.placeholder-note {{ font-size: 11px; color: #b5859a; font-style: italic; margin-top: 2px; }}
.app-title {{ font-weight: 800; letter-spacing: -0.01em; }}
.app-subtitle {{ color: #666; }}
.byline {{ color: #999; font-size: 13px; margin-top: -6px; }}
div[data-testid="stSidebar"] {{ background: #fafafa; border-right: 1px solid #eee; }}
.metric-glossary dt {{ font-weight: 700; margin-top: 8px; }}
.metric-glossary dd {{ margin-left: 0; color: #444; margin-bottom: 4px; }}

/* Primary button (Run classification) recoloured pink */
.stButton button[kind="primary"] {{
    background-color: {ACCENT} !important;
    border-color: {ACCENT} !important;
    color: white !important;
}}
.stButton button[kind="primary"]:hover {{
    background-color: #a01070 !important;
    border-color: #a01070 !important;
}}

/* File uploader accent elements recoloured pink */
[data-testid="stFileChip"] svg {{ color: {ACCENT} !important; fill: {ACCENT} !important; }}
[data-testid="stFileUploaderDropzone"] svg {{ color: {ACCENT} !important; fill: {ACCENT} !important; }}
section[data-testid="stFileUploaderDropzone"] {{ border-color: {ACCENT}55 !important; }}

/* Uploaded-file icon badge: white background, black border, pink icon.
   Confirmed via browser inspector: the real element is [data-testid="stFileChip"],
   with the icon svg sitting inside its first child div. */
[data-testid="stFileChip"] > div:first-child {{
    background: white !important;
    border: 1.5px solid #1a1a1a !important;
    border-radius: 7px !important;
}}
[data-testid="stFileChip"] svg {{
    color: {ACCENT} !important;
    fill: {ACCENT} !important;
}}

/* Sidebar width — a bit wider so mode names don't truncate */
section[data-testid="stSidebar"] {{ width: 380px !important; }}
section[data-testid="stSidebar"] > div:first-child {{ width: 380px !important; }}

/* Hover-info icon + popover, replaces the click-to-expand stats glossary */
.info-tooltip-wrap {{
    position: relative; display: inline-block; cursor: help; margin: 10px 0 4px 0;
}}
.info-icon {{
    font-weight: 600; color: {ACCENT}; border: 1.5px solid {ACCENT}; border-radius: 999px;
    padding: 5px 14px; font-size: 14px; background: white; display: inline-block;
}}
.info-tooltip-content {{
    display: none; position: absolute; top: 130%; left: 0; z-index: 999;
    background: white; border: 1px solid #eee; border-radius: 10px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.14); padding: 18px 22px;
    width: 480px; max-height: 440px; overflow-y: auto; cursor: default;
}}
/* Small inline hover-info icon for individual stat cards */
.info-icon-sm {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 15px; height: 15px; border-radius: 50%; border: 1.3px solid {ACCENT};
    color: {ACCENT}; font-size: 10px; font-weight: 700; margin-left: 5px;
    cursor: help; vertical-align: middle;
}}
.info-tooltip-wrap-sm {{ position: relative; display: inline-block; }}
.info-tooltip-content-sm {{
    display: none; position: absolute; top: 130%; left: 50%; transform: translateX(-50%);
    z-index: 999; background: #1a1a1a; color: white; border-radius: 8px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.2); padding: 10px 13px;
    width: 210px; font-size: 12px; font-weight: 400; text-align: left; line-height: 1.4;
    cursor: default;
}}
.info-tooltip-wrap-sm:hover .info-tooltip-content-sm {{ display: block; }}

.section-toggle summary {{
    font-size: 1.5rem; font-weight: 700; color: rgb(49, 51, 63);
    cursor: pointer; padding: 4px 0; list-style: revert;
}}
.section-toggle summary:hover {{ color: {ACCENT}; }}
.section-toggle .section-body {{ padding: 8px 2px 4px 2px; }}
.section-toggle ol {{ margin: 0; padding-left: 22px; }}
.section-toggle li {{ margin-bottom: 8px; }}
code {{
    background-color: #fdf0f6 !important; color: {ACCENT} !important;
    padding: 2px 6px; border-radius: 4px;
}}
/* Progress bar fill, in case the theme config doesn't cover it */
div[data-testid="stProgress"] div[role="progressbar"] > div,
div[data-testid="stProgress"] > div > div > div {{
    background-color: {ACCENT} !important;
}}
/* Focus ring on text inputs and selectboxes — red by default, recoloured pink */
div[data-baseweb="select"]:focus-within > div,
div[data-baseweb="base-input"]:focus-within,
div[data-baseweb="input"]:focus-within {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 1px {ACCENT} !important;
}}
input:focus, textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 1px {ACCENT} !important;
    outline-color: {ACCENT} !important;
}}
</style>
""", unsafe_allow_html=True)

BADGE_CLASSES = ["badge-baseline", "badge-c1", "badge-c2"]


def badge_html(label, class_idx):
    cls = BADGE_CLASSES[min(class_idx, len(BADGE_CLASSES) - 1)]
    return f'<span class="badge {cls}">{label}</span>'


def stat_card(label, value, placeholder=False, info=None):
    note = '<div class="placeholder-note">provisional definition</div>' if placeholder else ""
    icon = ""
    if info:
        icon = (
            f'<span class="info-tooltip-wrap-sm"><span class="info-icon-sm">i</span>'
            f'<div class="info-tooltip-content-sm">{info}</div></span>'
        )
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">{label}{icon}</div>'
        f'<div class="stat-value">{value}</div>{note}</div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
# --- EASILY EDITABLE: title, subtitle, byline, and logo live here ---
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.markdown('<div class="app-title" style="font-size:38px;">FSCV Serotonin Release Event Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle" style="font-size:16px;">Hashemi Lab — Automated Classification of Serotonin Release Events from FSCV Recordings</div>', unsafe_allow_html=True)
    st.markdown('<div class="byline">Built by Juliette Desbiez · MSc Bioengineering, Imperial College London</div>', unsafe_allow_html=True)
with header_col2:
    st.write("")
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
st.write("")

# ------------------------------------------------------------------
# Session state defaults
# ------------------------------------------------------------------
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0


def do_restart():
    st.session_state["uploader_key"] += 1
    for k in ("results", "results_bundle", "results_labels"):
        st.session_state.pop(k, None)


# ------------------------------------------------------------------
# Sidebar: bundle selection + display label editor
# ------------------------------------------------------------------
with st.sidebar:
    st.header("1 · Choose a mode")
    bundle_key = st.selectbox(
        "Preparation & classification mode",
        options=list(BUNDLES.keys()),
        format_func=lambda k: BUNDLES[k]["menu_label"],
    )
    bundle = BUNDLES[bundle_key]
    st.caption(f"Classes: {' / '.join(bundle.get('default_display_names', bundle['class_names']))}")
    st.caption(f"Held-out test F1_macro = {bundle['test_f1_macro']:.4f}")
    model_desc = "Model: MLP (single model)" if bundle["model_type"] == "mlp_only" else "Model: Ensemble (RF + XGBoost + MLP)"
    st.caption(model_desc)

    st.header("2 · Display labels")
    st.caption("These labels only change what's shown on screen — the model's underlying classification is unaffected.")

    default_labels = bundle.get("default_display_names", bundle["class_names"])
    if "label_state" not in st.session_state or st.session_state.get("label_bundle") != bundle_key:
        st.session_state["label_state"] = list(default_labels)
        st.session_state["label_bundle"] = bundle_key

    new_labels = []
    for i in range(bundle["n_classes"]):
        val = st.text_input(f"Class {i} label", value=st.session_state["label_state"][i], key=f"label_{bundle_key}_{i}")
        new_labels.append(val.strip())
    st.session_state["label_state"] = new_labels

    label_error = None
    if any(l == "" for l in new_labels):
        label_error = "Labels cannot be blank."
    elif len(set(new_labels)) != len(new_labels):
        label_error = "Labels must be unique within a mode."
    if label_error:
        st.error(label_error)

    display_labels = new_labels if not label_error else default_labels

    st.header("3 · Upload recordings")
    uploaded_files = st.file_uploader(
        "Raw FSCV .txt recordings (batch upload supported)",
        type=["txt"], accept_multiple_files=True,
        key=f"uploader_{st.session_state['uploader_key']}",
    )
    run_clicked = st.button("Run classification", type="primary", disabled=bool(label_error) or not uploaded_files)
    st.button("↺ Restart (clear files & results)", on_click=do_restart)


# ------------------------------------------------------------------
# Home / instructions panel — shown until results exist
# ------------------------------------------------------------------
def render_instructions():
    how_it_works_html = """
<details class="section-toggle" open>
<summary>How this works</summary>
<div class="section-body">
<ol>
<li><strong>Choose a mode</strong> in the sidebar — this picks the preparation, the trained
model(s), and the number of classes. Nothing else needs configuring.</li>
<li><strong>Rename the display labels</strong> if you'd like different wording — this is
cosmetic only.</li>
<li><strong>Upload one or more raw <code>.txt</code> FSCV recordings.</strong> Batch upload is
supported.</li>
<li><strong>Click Run classification.</strong> Each recording is windowed, classified
window-by-window, and rolled up to one call per file using an <em>"any event override"</em>
rule: if even one window in a file is classified as an event, the whole file is called
non-baseline. A file is only called baseline if every window in it is.</li>
<li><strong>Review results</strong> — a summary table for the whole batch, plus a full detail
view (colour plot, stat cards, per-window breakdown) for any file you select.</li>
</ol>
</div>
</details>
"""
    st.markdown(how_it_works_html, unsafe_allow_html=True)


# ------------------------------------------------------------------
# Processing
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_models(bundle_key):
    return load_bundle_models(BUNDLES[bundle_key])


def process_file(file_obj, bundle, models):
    arr_raw = load_recording(file_obj)
    windows, spans, window_frames, arr_bg = window_recording(arr_raw, bundle["config"])
    if not windows:
        return None

    proba = predict_windows(bundle, models, windows)
    window_preds = np.argmax(proba, axis=1)
    file_class, confidence, rep_window_idx, positive_idx = aggregate_file(
        window_preds, proba, bundle["n_classes"]
    )

    features = None
    if rep_window_idx is not None:
        # Display-only sign correction, same principle as the colour plot:
        # classification above already ran on the original (non-inverted)
        # window, matching what the trained models learned from. This
        # separately re-extracts features from an inverted copy of just
        # that one representative window, purely so the stat cards show
        # physically correct values (e.g. a genuine oxidation peak reads
        # as positive) -- it has no effect on classification.
        features = extract(-windows[rep_window_idx], bundle["config"])

    event_spans = [spans[i] for i in positive_idx]

    return {
        "filename": getattr(file_obj, "name", "recording.txt"),
        "file_class": file_class,
        "confidence": confidence,
        "features": features,
        "n_event_windows": len(positive_idx),
        "n_windows": len(windows),
        "event_spans": event_spans,
        "arr": arr_bg,
        "window_preds": window_preds,
        "proba": proba,
        "spans": spans,
        "fscv_hz": bundle["config"]["fscv_hz"],
    }


content_area = st.empty()


if run_clicked and uploaded_files:
    models = get_models(bundle_key)
    results = []
    progress = st.progress(0.0, text="Processing recordings...")
    for i, f in enumerate(uploaded_files):
        r = process_file(f, bundle, models)
        if r is not None:
            results.append(r)
        progress.progress((i + 1) / len(uploaded_files))
    progress.empty()
    st.session_state["results"] = results
    st.session_state["results_bundle"] = bundle_key
    st.session_state["results_labels"] = display_labels


# ------------------------------------------------------------------
# Results display — Option C: summary table + expandable detail
# ------------------------------------------------------------------
FEATURE_CSV_ORDER = [
    ("peak_current", "amplitude"), ("rise_time", "rise_time"), ("decay_time", "decay_time"),
    ("peak_width", "peak_width_fwhm"), ("auc_sero", "auc_oxidation"), ("auc_full", "auc_full"),
    ("ox_red_ratio", "ox_red_ratio"), ("ox_red_lag", "ox_red_lag"),
]

with content_area.container():
    if "results" in st.session_state and st.session_state.get("results_bundle") == bundle_key:
        results = st.session_state["results"]
        labels = st.session_state["results_labels"]

        if not results:
            st.warning("No windows could be extracted from the uploaded file(s) — check they're long enough for a 2s window.")
        else:
            st.subheader(f"Results — {len(results)} recording(s)")

            summary_rows = []
            full_rows = []
            for r in results:
                summary_rows.append({
                    "File": r["filename"],
                    "Classification": labels[r["file_class"]],
                    "Confidence": f"{r['confidence']:.3f}",
                })
                row = {
                    "file": r["filename"],
                    "classification": labels[r["file_class"]],
                    "confidence": round(r["confidence"], 4),
                    "n_event_windows": r["n_event_windows"],
                    "n_total_windows": r["n_windows"],
                }
                for feat_key, col_name in FEATURE_CSV_ORDER:
                    row[col_name] = round(r["features"][feat_key], 4) if r["features"] else ""
                full_rows.append(row)

            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "Download batch summary CSV (simple)",
                    data=summary_df.to_csv(index=False).encode("utf-8"),
                    file_name="fscv_batch_summary.csv",
                    mime="text/csv",
                    help="File, classification, confidence — one row per recording.",
                )
            with col_b:
                full_df = pd.DataFrame(full_rows)
                st.download_button(
                    "Download full batch results CSV (with all stats)",
                    data=full_df.to_csv(index=False).encode("utf-8"),
                    file_name="fscv_batch_full_results.csv",
                    mime="text/csv",
                    help="Everything in the summary, plus amplitude, rise/decay time, peak width, "
                         "AUC, Ox/Red ratio and lag, and window counts for every recording.",
                )

            st.divider()
            st.subheader("File detail")
            filenames = [r["filename"] for r in results]
            selected_name = st.selectbox("Select a file to inspect", filenames)
            r = next(x for x in results if x["filename"] == selected_name)

            st.markdown(badge_html(labels[r["file_class"]], r["file_class"]), unsafe_allow_html=True)

            cols = st.columns(4)
            with cols[0]:
                stat_card("Classification", labels[r["file_class"]],
                          info="The file's overall predicted class, using your chosen display labels.")
            with cols[1]:
                stat_card("Confidence", f"{r['confidence']:.3f}",
                          info="The model's probability for the single most confident event window that decided the file's classification.")
            with cols[2]:
                amp = f"{r['features']['peak_current']:.1f} nA" if r["features"] else "—"
                stat_card("Amplitude", amp,
                          info="The peak oxidation current from the specific event window that determined the file's classification.")
            with cols[3]:
                rt = f"{r['features']['rise_time']:.0f} f" if r["features"] else "—"
                stat_card("Rise time", rt,
                          info="Frames from the start of the window to the oxidation peak.")

            cols2 = st.columns(4)
            with cols2[0]:
                dt = f"{r['features']['decay_time']:.2f}" if r["features"] else "—"
                stat_card("Decay time", dt,
                          info="Rate of signal decline per frame after the oxidation peak.")
            with cols2[1]:
                pw = f"{r['features']['peak_width']:.0f}" if r["features"] else "—"
                stat_card("Peak width (FWHM)", pw,
                          info="Width of the oxidation peak at half its maximum height.")
            with cols2[2]:
                auc = f"{r['features']['auc_sero']:.0f} / {r['features']['auc_full']:.0f}" if r["features"] else "—"
                stat_card("AUC (ox / full)", auc,
                          info="Area under the curve across the oxidation band, and across the whole window.")
            with cols2[3]:
                orl = f"{r['features']['ox_red_ratio']:.2f} / {r['features']['ox_red_lag']:.0f}" if r["features"] else "—"
                stat_card("Ox/Red ratio / lag", orl,
                          info="Ratio of the oxidation peak to the reduction trough, and the frame gap between the oxidation peak and reduction trough.")

            # --- Colour plot with event-window track ---
            st.markdown("#### Colour plot")
            arr = r["arr"]
            nV, nT = arr.shape
            fscv_hz = r["fscv_hz"]
            max_t = (nT - 1) / fscv_hz

            # Display-only sign inversion: the raw .txt files (and every
            # training/labelling file used to build the current models) use
            # a sign convention that renders oxidation as blue and reduction
            # as green on Pablo's colormap -- backwards from the physically
            # expected polarity. This flips ONLY the array handed to imshow;
            # `arr` itself (used above for classification) is untouched, so
            # this has zero effect on windowing, features, or predictions --
            # it only corrects what's drawn on screen.
            arr_display = -arr
            norm = PLOT_SETTINGS.get_norm(arr_display)

            fig, (ax, ax_track) = plt.subplots(
                2, 1, figsize=(11, 5.2), dpi=150,
                gridspec_kw={"height_ratios": [5, 0.6], "hspace": 0.06},
                sharex=True,
            )
            ax.imshow(arr_display, aspect="auto", cmap=PLOT_SETTINGS.custom, origin="lower",
                       extent=[0, max_t, 0, nV], norm=norm)

            for (f0, f1) in r["event_spans"]:
                ax.axvspan(f0 / fscv_hz, f1 / fscv_hz, color=ACCENT, alpha=0.25, lw=0)

            # Real voltage values on the y-axis instead of raw row index.
            # The Jackson waveform is non-monotonic (rises 0.2V->1.0V, falls
            # to -0.1V, rises back to 0.2V), so this can't be a simple linear
            # relabel — each tick is placed at its actual row and labelled
            # with the true voltage measured at that row.
            n_ticks = 8
            tick_rows = np.linspace(0, nV - 1, n_ticks).astype(int)
            tick_rows = np.clip(tick_rows, 0, len(VOLTAGE_VALUES) - 1)
            ax.set_yticks(tick_rows)
            ax.set_yticklabels([f"{VOLTAGE_VALUES[i]:.2f}" for i in tick_rows])
            ax.set_ylabel("Voltage (V)")
            ax.set_title(f"{r['filename']} \u2014 {len(r['event_spans'])} event window(s) highlighted")
            ax.tick_params(labelbottom=False)

            ax_track.set_xlim(0, max_t)
            ax_track.set_ylim(0, 1)
            ax_track.set_yticks([])
            ax_track.set_facecolor("#f2f2f2")
            for (f0, f1) in r["event_spans"]:
                ax_track.axvspan(f0 / fscv_hz, f1 / fscv_hz, color=ACCENT_BAR, alpha=0.9, lw=0)
            ax_track.set_xlabel("Time (s)")
            ax_track.set_ylabel("Event\nwindows", fontsize=8, rotation=0, ha="right", va="center")

            st.pyplot(fig)
            plt.close(fig)
            if r["event_spans"]:
                st.caption("Translucent pink shading marks event windows on the colour plot; "
                           "the pink track below shows event windows independent of plot colour.")

            with st.expander("Per-window predictions"):
                win_rows = []
                for i, (f0, f1) in enumerate(r["spans"]):
                    win_rows.append({
                        "start_s": f0 / r["fscv_hz"],
                        "end_s": f1 / r["fscv_hz"],
                        "predicted_class": labels[r["window_preds"][i]],
                        **{f"proba_{labels[c]}": r["proba"][i, c] for c in range(len(labels))},
                    })
                win_df = pd.DataFrame(win_rows)
                st.dataframe(win_df, use_container_width=True, hide_index=True)
                st.download_button(
                    f"Download {selected_name} per-window CSV",
                    data=win_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected_name}_windows.csv",
                    mime="text/csv",
                )
    else:
        render_instructions()
