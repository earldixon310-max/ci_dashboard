# app.py
import streamlit as st
import pandas as pd


# IMPORTANT: set_page_config must be the first Streamlit call
st.set_page_config(page_title="CI Dashboard", layout="wide")

from lib.ci_io import load_report, available_tiles, select_bundle
from lib.ci_math import windows_to_df, compute_conf_from_schema, conf_summary, tier_from_ci
from lib.plots import plot_valid_pie, plot_delta_hist, plot_corr_delta_scatter
from lib.ci_ui import inject_reference_css, render_tier_status_card
from lib.ci_ui import inject_reference_css, render_ci_summary_block

# Inject the reference look & feel once
inject_reference_css("CI Dashboard")



st.title("CI Dashboard")
st.caption("Operator view — deterministic CI metrics (no manual tuning).")

# Load report
try:
    report = load_report("ci_report.json")
except Exception as e:
    st.error(f"Failed to load ci_report.json: {e}")
    st.stop()

# Scope picker
tiles = available_tiles(report)
tile_options = ["GLOBAL"] + [t["id"] for t in tiles]
tile_pick = st.selectbox(
    "Scope",
    tile_options,
    index=0,
    help="GLOBAL = metrics computed across ALL windows (all tiles combined). Tiles = metrics for that tile only."
)


tile_id = None if tile_pick == "GLOBAL" else tile_pick
bundle = select_bundle(report, tile_id=tile_id)

# --------------------------
# Alerts "state" (read-only)
# --------------------------
# Set these from your real feed code when available.
# For now, keep them resilient and simple.
alerts_state = {
    "nws_ok": False,
    "nhc_ok": False,
    "nws_count": 0,
    "nhc_count": 0,
    "hazard_elevated": False,
    "note": "",
}

# Example: if your feed call failed, treat as "unknown", not "no hazards"
# alerts_state["note"] = "NWS feed unreachable (DNS)."

# Pull metrics (prefer bundle metrics; fall back to zeros)
m = bundle.get("metrics", {}) or {}
ci_conf = float(m.get("ci_confidence") or 0.0)
presence = float(m.get("presence") or 0.0)
stability = float(m.get("stability") or 0.0)
reliability = float(m.get("reliability") or 0.0)

tier_name, tier_code = tier_from_ci(ci_conf)
explanation = (bundle.get("interpretation", {}) or {}).get("explanation") or ""

# Windows DF + confidence summary
dfw = windows_to_df(bundle.get("windows", []) or [])
df_conf = compute_conf_from_schema(dfw)
cs = conf_summary(df_conf)

# ----- GLOBAL decomposition: tile contributions -----
tiles_list = available_tiles(report)  # list of tile dicts
rows = []

total_windows = 0
for t in tiles_list:
    counts = t.get("counts", {}) or {}
    total_windows += int(counts.get("N_windows") or 0)

for t in tiles_list:
    tid = t.get("id")
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

best_tile_row = None
if not df_tiles.empty:
    best_tile_row = df_tiles.iloc[0].to_dict()



weighted_ci_sum = float(df_tiles["weighted_ci_share"].sum()) if not df_tiles.empty else 0.0

left, right = st.columns([1.0, 1.0], gap="large")

c1, c2, c3 = st.columns(3, gap="small")
with c1:
    plot_valid_pie(dfw, title="Valid vs Invalid", figsize=(2.6, 2.2))
with c2:
    plot_delta_hist(dfw, title="Δ (valid)", figsize=(2.6, 2.2), bins=8)
with c3:
    plot_corr_delta_scatter(dfw, title="|corr| vs Δ", figsize=(2.6, 2.2))

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
    
 # ✅ Read-only context badge (does not modify CI)
    ext_count = alerts_state["nws_count"] + alerts_state["nhc_count"]
    if alerts_state["hazard_elevated"]:
        st.caption(f"🛈 Context: Active NWS/NHC alerts present ({ext_count}) — external signal (not part of CI).")
    elif (not alerts_state["nws_ok"]) or (not alerts_state["nhc_ok"]):
        st.caption("🛈 Context: Alerts status unknown (feed unavailable) — external signal (not part of CI).")
    else:
        st.caption("🛈 Context: No active NWS/NHC alerts detected — external signal (not part of CI).")
 # ✅ Interpretation banner (safe: interpretation only)
if tier_code == "exploratory":
    if alerts_state["hazard_elevated"]:
        st.warning("CI is Exploratory. External hazard activity is elevated — interpret correlations cautiously.")
    else:
        st.info("CI is Exploratory. Use for discovery only — do not treat as operational guidance.")
elif tier_code == "conditional":
    if alerts_state["hazard_elevated"]:
        st.info("CI is Conditional. External hazard activity is elevated — treat CI as supportive context, not a trigger.")
else:
    # admissible/unknown
    if alerts_state["hazard_elevated"]:
        st.success("CI is strong. External hazard activity is elevated — CI may help prioritize monitoring focus.")
       
st.markdown('<hr class="sep"/>', unsafe_allow_html=True)
st.subheader("GLOBAL vs strongest tile")

if tier_code == "exploratory":
    st.caption("Situational awareness only: CI gating not met — alerts shown for context, not decision support.")
elif tier_code == "conditional":
    st.caption("Alerts shown for operational context. CI is supportive but not decisive.")
else:
    st.caption("Alerts shown normally (CI is strong enough to be used as decision support context).")


if df_tiles is None or df_tiles.empty:
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


    st.markdown('<hr class="sep"/>', unsafe_allow_html=True)

    st.subheader("Confidence (window-level)")
    if cs["n"] == 0 or cs["confidence_median"] is None:
        st.info("No windows available for confidence scoring.")
    else:
        st.metric("Median confidence", f"{cs['confidence_median']:.2f}")
        st.metric("High-confidence windows", f"{100*cs['high_frac']:.0f}%")

"""
with right:
    st.subheader("Quick charts")
    plot_valid_pie(dfw)
    plot_delta_hist(dfw)
    plot_corr_delta_scatter(dfw)
"""
st.markdown('<hr class="sep"/>', unsafe_allow_html=True)

st.subheader("Window summary (single table)")
if df_conf is None or df_conf.empty:
    st.info("No windows available.")
else:
    show_cols = [
        "index","start","end","valid","tau_star","corr","delta","validity_reason","tile_id",
        "confidence_label","confidence"
    ]
    show_cols = [c for c in show_cols if c in df_conf.columns]
    st.dataframe(
        df_conf[show_cols].sort_values(["valid","confidence"], ascending=[False, False]),
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("GLOBAL decomposition (tile contributions)")
if df_tiles.empty or total_windows == 0:
    st.info("No tile windows available to decompose GLOBAL.")
else:
    st.caption(f"Weighted CI sum (Σ weight·tile_ci) = {weighted_ci_sum:.3f}  |  Reported GLOBAL ci_confidence = {ci_conf:.3f}")
    st.dataframe(
        df_tiles[[
            "tile_id","N_windows","N_valid","weight",
            "tile_ci","tile_presence","tile_stability","tile_reliability",
            "weighted_ci_share"
        ]],
        use_container_width=True,
        hide_index=True
    )

