# app.py
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# -----------------------------
# Streamlit config MUST be first
# -----------------------------
st.set_page_config(page_title="CI Dashboard", layout="wide")

# -----------------------------
# Ensure project root is importable (critical for /mount/src deployments)
# -----------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -----------------------------
# Local imports
# -----------------------------
from lib.ci_io import load_report, available_tiles, select_bundle
from lib.ci_math import windows_to_df, compute_conf_from_schema, conf_summary, tier_from_ci
from lib.plots import plot_valid_pie, plot_delta_hist, plot_corr_delta_scatter
from lib.ci_ui import inject_reference_css, render_tier_status_card, render_ci_summary_block

# -----------------------------
# Style
# -----------------------------
inject_reference_css("CI Dashboard")

st.title("CI Dashboard")
st.caption("Operator view — deterministic CI metrics (no manual tuning).")

# -----------------------------
# Load report (prefer runtime file, fall back to example)
# -----------------------------
PRIMARY = Path("ci_report.json")
FALLBACK = Path("ci_report.example.json")

report_path = PRIMARY if PRIMARY.exists() else FALLBACK

try:
    report = load_report(str(report_path))
except Exception as e:
    st.error(f"Failed to load {report_path.name}: {e}")
    st.stop()

# Guard: report must be dict
if not isinstance(report, dict):
    st.error(f"Report must be a JSON object (dict), got {type(report)}")
    st.stop()

st.caption(f"Loaded: `{report_path.name}`")

# -----------------------------
# Scope picker (GLOBAL vs tile)
# -----------------------------
tiles_raw = available_tiles(report) or []

# Support both shapes:
# - list[dict] tiles (preferred)
# - list[str] tile ids
tile_dicts = []
tile_ids = []

if tiles_raw:
    if isinstance(tiles_raw[0], dict):
        tile_dicts = tiles_raw
        tile_ids = [str(t.get("id")) for t in tile_dicts if t.get("id") is not None]
    else:
        # assume list of IDs
        tile_ids = [str(x) for x in tiles_raw]

tile_options = ["GLOBAL"] + tile_ids

tile_pick = st.selectbox(
    "Scope",
    tile_options,
    index=0,
    help="GLOBAL = metrics computed across ALL windows (all tiles combined). Tiles = metrics for that tile only.",
)

tile_id = None if tile_pick == "GLOBAL" else tile_pick
bundle = select_bundle(report, tile_id=tile_id)

# -----------------------------
# Alerts state (read-only context)
# -----------------------------
alerts_state = {
    "nws_ok": False,
    "nhc_ok": False,
    "nws_count": 0,
    "nhc_count": 0,
    "hazard_elevated": False,
    "note": "",
}

# -----------------------------
# Pull metrics
# -----------------------------
m = (bundle.get("metrics", {}) or {}) if isinstance(bundle, dict) else {}
ci_conf = float(m.get("ci_confidence") or 0.0)
presence = float(m.get("presence") or 0.0)
stability = float(m.get("stability") or 0.0)
reliability = float(m.get("reliability") or 0.0)

tier_name, tier_code = tier_from_ci(ci_conf)
explanation = ((bundle.get("interpretation", {}) or {}).get("explanation") or "") if isinstance(bundle, dict) else ""

# -----------------------------
# Windows DF + confidence
# -----------------------------
windows = (bundle.get("windows", []) or []) if isinstance(bundle, dict) else []
dfw = windows_to_df(windows)
df_conf = compute_conf_from_schema(dfw)
cs = conf_summary(df_conf)

# -----------------------------
# GLOBAL decomposition (tile contributions)
# Only works if we have tile dicts with counts/metrics
# -----------------------------
rows = []
total_windows = 0

if tile_dicts:
    for t in tile_dicts:
        counts = t.get("counts", {}) or {}
        total_windows += int(counts.get("N_windows") or 0)

    for t in tile_dicts:
        tid = str(t.get("id"))
        counts = t.get("counts", {}) or {}
        metrics = t.get("metrics", {}) or {}

        nW = int(counts.get("N_windows") or 0)
        nV = int(counts.get("N_valid") or 0)
        weight = (nW / total_windows) if total_windows > 0 else 0.0

        t_ci = float(metrics.get("ci_confidence") or 0.0)
        t_p  = float(metrics.get("presence") or 0.0)
        t_s  = float(metrics.get("stability") or 0.0)
        t_r  = float(metrics.get("reliability") or 0.0)

        rows.append({
            "tile_id": tid,
            "N_windows": nW,
            "N_valid": nV,
            "weight": weight,
            "tile_ci": t_ci,
            "tile_presence": t_p,
            "tile_stability": t_s,
            "tile_reliability": t_r,
            "weighted_ci_share": weight * t_ci,
        })

df_tiles = pd.DataFrame(rows)
if not df_tiles.empty:
    df_tiles = df_tiles.sort_values(["tile_ci", "N_windows"], ascending=[False, False])

weighted_ci_sum = float(df_tiles["weighted_ci_share"].sum()) if not df_tiles.empty else 0.0

# -----------------------------
# Top row charts
# -----------------------------
c1, c2, c3 = st.columns(3, gap="small")
with c1:
    plot_valid_pie(dfw, title="Valid vs Invalid", figsize=(2.6, 2.2))
with c2:
    plot_delta_hist(dfw, title="Δ (valid)", figsize=(2.6, 2.2), bins=8)
with c3:
    plot_corr_delta_scatter(dfw, title="|corr| vs Δ", figsize=(2.6, 2.2))

# -----------------------------
# Status + Interpretation (left column)
# -----------------------------
left, right = st.columns([1.0, 1.0], gap="large")

with left:
    st.subheader("Status")

    render_tier_status_card(
        tier_name=tier_name,
        tier_code=tier_code,
        ci_conf=ci_conf,
        presence=presence,
        stability=stability,
        reliability=reliability,
        explanation=explanation,
    )

    # Optional summary block (if you intended to use it)
    try:
        render_ci_summary_block(bundle)
    except Exception:
        pass

    # Read-only context badge (does not modify CI)
    ext_count = int(alerts_state["nws_count"]) + int(alerts_state["nhc_count"])
    if alerts_state["hazard_elevated"]:
        st.caption(f"🛈 Context: Active NWS/NHC alerts present ({ext_count}) — external signal (not part of CI).")
    elif (not alerts_state["nws_ok"]) or (not alerts_state["nhc_ok"]):
        st.caption("🛈 Context: Alerts status unknown (feed unavailable) — external signal (not part of CI).")
    else:
        st.caption("🛈 Context: No active NWS/NHC alerts detected — external signal (not part of CI).")

    # Interpretation banner (safe: interpretation only)
    if tier_code == "exploratory":
        if alerts_state["hazard_elevated"]:
            st.warning("CI is Exploratory. External hazard activity is elevated — interpret correlations cautiously.")
        else:
            st.info("CI is Exploratory. Use for discovery only — do not treat as operational guidance.")
    elif tier_code == "conditional":
        if alerts_state["hazard_elevated"]:
            st.info("CI is Conditional. External hazard activity is elevated — treat CI as supportive context, not a trigger.")
        else:
            st.info("CI is Conditional. CI is supportive but not decisive.")
    else:
        # admissible / unknown
        if alerts_state["hazard_elevated"]:
            st.success("CI is strong. External hazard activity is elevated — CI may help prioritize monitoring focus.")

# -----------------------------
# GLOBAL vs strongest tile
# -----------------------------
st.markdown('<hr class="sep"/>', unsafe_allow_html=True)
st.subheader("GLOBAL vs strongest tile")

if tier_code == "exploratory":
    st.caption("Situational awareness only: CI gating not met — alerts shown for context, not decision support.")
elif tier_code == "conditional":
    st.caption("Alerts shown for operational context. CI is supportive but not decisive.")
else:
    st.caption("Alerts shown normally (CI is strong enough to be used as decision support context).")

if df_tiles.empty:
    st.info("No tile data available.")
else:
    best = df_tiles.iloc[0].to_dict()
    colA, colB = st.columns(2, gap="large")

    with colA:
        st.markdown(
            f"""
<div class="ci-card">
  <h3>GLOBAL</h3>
  <div class="metric-grid">
    <div class="metric-tile"><div class="metric-label">CI Confidence</div><div class="metric-value">{ci_conf:.3f}</div></div>
    <div class="metric-tile"><div class="metric-label">Presence</div><div class="metric-value">{presence:.3f}</div></div>
    <div class="metric-tile"><div class="metric-label">Stability</div><div class="metric-value">{stability:.3f}</div></div>
    <div class="metric-tile"><div class="metric-label">Reliability</div><div class="metric-value">{reliability:.3f}</div></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with colB:
        st.markdown(
            f"""
<div class="ci-card">
  <h3>Strongest tile: {best['tile_id']}</h3>
  <div class="metric-grid">
    <div class="metric-tile"><div class="metric-label">CI Confidence</div><div class="metric-value">{float(best['tile_ci']):.3f}</div></div>
    <div class="metric-tile"><div class="metric-label">Presence</div><div class="metric-value">{float(best['tile_presence']):.3f}</div></div>
    <div class="metric-tile"><div class="metric-label">Stability</div><div class="metric-value">{float(best['tile_stability']):.3f}</div></div>
    <div class="metric-tile"><div class="metric-label">Reliability</div><div class="metric-value">{float(best['tile_reliability']):.3f}</div></div>
  </div>
  <div class="small-muted" style="margin-top:10px;">
    Weight (by windows): {float(best['weight']):.3f} • N_windows={int(best['N_windows'])} • N_valid={int(best['N_valid'])}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

# -----------------------------
# Confidence summary
# -----------------------------
st.markdown('<hr class="sep"/>', unsafe_allow_html=True)
st.subheader("Confidence (window-level)")

if cs.get("n", 0) == 0 or cs.get("confidence_median") is None:
    st.info("No windows available for confidence scoring.")
else:
    st.metric("Median confidence", f"{cs['confidence_median']:.2f}")
    st.metric("High-confidence windows", f"{100*cs['high_frac']:.0f}%")

# -----------------------------
# Window table
# -----------------------------
st.markdown('<hr class="sep"/>', unsafe_allow_html=True)
st.subheader("Window summary (single table)")

if df_conf is None or df_conf.empty:
    st.info("No windows available.")
else:
    show_cols = [
        "index", "start", "end", "valid", "tau_star", "corr", "delta",
        "validity_reason", "tile_id", "confidence_label", "confidence"
    ]
    show_cols = [c for c in show_cols if c in df_conf.columns]

    st.dataframe(
        df_conf[show_cols].sort_values(["valid", "confidence"], ascending=[False, False]),
        use_container_width=True,
        hide_index=True,
    )

# -----------------------------
# GLOBAL decomposition table
# -----------------------------
st.subheader("GLOBAL decomposition (tile contributions)")

if df_tiles.empty or total_windows == 0:
    st.info("No tile windows available to decompose GLOBAL.")
else:
    st.caption(
        f"Weighted CI sum (Σ weight·tile_ci) = {weighted_ci_sum:.3f}  |  "
        f"Reported GLOBAL ci_confidence = {ci_conf:.3f}"
    )
    st.dataframe(
        df_tiles[
            [
                "tile_id", "N_windows", "N_valid", "weight",
                "tile_ci", "tile_presence", "tile_stability", "tile_reliability",
                "weighted_ci_share",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
