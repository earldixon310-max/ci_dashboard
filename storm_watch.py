# storm_watch.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import shutil
import pandas as pd
import streamlit as st
from storm_watch_core import (
    ensure_sw_shape,
    load_json,
    save_json,
    latest_update,
    checkpoint_rows,
    compute_grade_from_update,
    confidence_trend,
    improvement_hints,
    compute_p_mix,
    simulate_next_update,
    append_update,
    )

# -------------------------
# Public API (imported by app.py)
# -------------------------
def load_storm_watch(path: Path) -> Dict[str, Any]:
    """Load storm watch json and normalize to required shape."""
    sw = load_json(path)
    return ensure_sw_shape(sw)

def demo_reset_button(example_path: Path, live_path: Path):
    if st.button("🔁 Reset demo to starting state"):
        if example_path.exists():
            shutil.copy(example_path, live_path)
            st.success("Demo reset to example baseline.")
            st.rerun()
        else:
            st.error("storm_watch.example.json not found.")

from typing import Any, Dict, Optional, List
import math

def _pmix(u: Dict[str, Any]) -> Optional[float]:
    obs = (u.get("observations", {}) or {})
    pp = (obs.get("precip_phase_probability") or {})
    peak = pp.get("peak") or {}
    if not peak:
        return None
    try:
        v = compute_p_mix(peak)
        fv = float(v)
        return fv if not math.isnan(fv) else None
    except Exception:
        return None


def _peakwin(u: Dict[str, Any]) -> Optional[float]:
    obs = (u.get("observations", {}) or {})
    tu = (obs.get("timing_uncertainty") or {})
    v = tu.get("peak_window_h")
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def what_changed(prev: Optional[Dict[str, Any]], cur: Dict[str, Any]) -> str:
    """Short change narrative between consecutive updates."""
    if not prev:
        return "First logged update."

    pm_prev, pm_cur = _pmix(prev), _pmix(cur)
    pw_prev, pw_cur = _peakwin(prev), _peakwin(cur)

    parts: List[str] = []

    if pm_prev is not None and pm_cur is not None:
        d = pm_cur - pm_prev
        if d > 0.02:
            parts.append("Mix risk ↑")
        elif d < -0.02:
            parts.append("Mix risk ↓")
        else:
            parts.append("Mix risk ~")

    if pw_prev is not None and pw_cur is not None:
        d = pw_cur - pw_prev
        if d > 0.5:
            parts.append("Peak window widened")
        elif d < -0.5:
            parts.append("Peak window tightened")
        else:
            parts.append("Peak window ~")

    return "; ".join(parts) if parts else "Update captured."


def render_confidence_log(sw: Dict[str, Any]) -> None:
    import streamlit as st
    import pandas as pd
    import math

    st.markdown("### Confidence Log (All Updates)")

    updates = sw.get("updates", []) or []
    if not updates:
        st.write("No update history.")
        return

    rows = []
    prev = None

    for urow in updates:
        grade_i, diag_i = compute_grade_from_update(urow)

        obs_i = (urow.get("observations", {}) or {})
        tu_i = (obs_i.get("timing_uncertainty") or {})

        pp_i = (obs_i.get("precip_phase_probability") or {})
        peak_i = pp_i.get("peak") or {}
        pmix_i = compute_p_mix(peak_i) if peak_i else float("nan")


        rows.append({
            "Time": urow.get("timestamp_iso"),
            "T-hrs": urow.get("t_minus_h"),
            "Grade": grade_i,
            "Mix risk (p_mix)": None if math.isnan(pmix_i) else round(pmix_i, 2),
            "Peak window (h)": tu_i.get("peak_window_h"),
            "What changed": what_changed(prev, urow),
        })
        prev = urow

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def render_storm_watch(sw: Dict[str, Any], live_path: Optional[Path] = None) -> None:
    import streamlit as st
    import altair as alt
    import math

    sw = ensure_sw_shape(sw)

    st.title("Storm Watch Mode")
    st.caption("Operator view — confidence-first storm tracking (no deterministic totals).")

    # ---- Demo controls (single expander) ----
    with st.expander("🧪 Demo Controls", expanded=False):
        colA, colB, colC = st.columns([1, 1, 1], gap="medium")

    with colA:
        if st.button("Simulate next update (demo)", type="primary"):
            u_new = simulate_next_update(sw)
            sw = append_update(sw, u_new)
            if live_path is not None:
                save_json(live_path, sw)
            st.rerun()

    with colB:
        if st.button("Update from CI (real)"):
            from storm_watch_update_from_ci import append_from_ci  # script function
            if live_path is None:
                st.error("No live_path available to save updates.")
            else:
                append_from_ci(
                    ci_path=Path("ci_report.json"),
                    sw_path=live_path,
                )
                st.success("Appended update from CI.")
                st.rerun()

    with colC:
        demo_reset_button(
            example_path=Path("storm_watch.example.json"),
            live_path=Path("storm_watch.json"),
        )



    # ---- Header ----
    target = sw.get("target", {}) or {}
    event = sw.get("event", {}) or {}
    zone = target.get("zone_id", "—")
    name = target.get("name", "—")
    st.subheader(f"{name} ({zone})")
    st.info("Context: External NWS products active (read-only). Alerts inform interpretation, not CI status.")

    ww = (event.get("watch_window") or {})
    if ww:
        st.caption(
            f"Watch window: {ww.get('start_iso','?')} → {ww.get('end_iso','?')} "
            f"(source: {ww.get('source','')})"
        )

    # ---- Storm Clock ----
    st.markdown("### Storm Clock (T-72 → T-0)")
    clock = checkpoint_rows(sw)
    if clock:
        df_clock = pd.DataFrame(clock)
        df_clock["drop"] = df_clock["drop"].apply(lambda x: "●" if x else "")
        st.dataframe(df_clock, use_container_width=True, hide_index=True)
    else:
        st.warning("No storm watch updates found in the file.")
        return

    u = latest_update(sw)
    if not u:
        st.warning("No latest update available.")
        return

    obs = u.get("observations", {}) or {}
    der = u.get("derived", {}) or {}
    trig = u.get("trigger_status", {}) or {}

    # ---- Confidence trajectory chart ----
    st.markdown("### Confidence trajectory (grades over time)")
    updates = sw.get("updates", []) or []
    if updates:
        gmap = {"A": 4, "B": 3, "C": 2, "D": 1}
        rows = []
        for uu in updates:
            g, diag = compute_grade_from_update(uu)
            rows.append({
                "timestamp": uu.get("timestamp_iso"),
                "t_minus_h": uu.get("t_minus_h"),
                "grade": g,
                "score": gmap.get(g),
                "p_mix": diag.get("p_mix"),
                "peak_window_h": diag.get("peak_window_h"),
                "onset_window_h": diag.get("onset_window_h"),
            })
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        use_time = df["timestamp"].notna().all()
        if use_time:
            df = df.sort_values("timestamp")
            x = alt.X("timestamp:T", title=None)
        else:
            df = df.sort_values("t_minus_h")
            x = alt.X("t_minus_h:Q", title="T-hrs")

        base = alt.Chart(df).properties(height=90)
        line = base.mark_line(point=False).encode(
            x=x,
            y=alt.Y("score:Q", title=None, scale=alt.Scale(domain=[1, 4])),
            tooltip=[
                alt.Tooltip("grade:N", title="Grade"),
                alt.Tooltip("p_mix:Q", format=".2f", title="p_mix"),
                alt.Tooltip("peak_window_h:Q", format=".0f", title="Peak win (h)"),
                alt.Tooltip("onset_window_h:Q", format=".0f", title="Onset win (h)"),
            ],
        )
        pts = base.mark_point(size=90).encode(x=x, y="score:Q")
        lbl = base.mark_text(dy=-12).encode(x=x, y="score:Q", text="grade:N")
        st.altair_chart(line + pts + lbl, use_container_width=True)
        st.caption("Grade scale: A (high confidence) → D (low confidence). Hover a point to see what drove the grade.")

    # ---- Snapshot ----
    st.markdown("### Current Snapshot")
    col1, col2, col3 = st.columns([1.0, 1.6, 1.0], gap="large")

    # A) Temperature tendency
    tt = (obs.get("temperature_tendency") or {})
    with col1:
        st.markdown("**Temperature Tendency**")
        st.write(f"ΔT 6h: {tt.get('dT_6h_C','—')} °C")
        st.write(f"ΔT 12h: {tt.get('dT_12h_C','—')} °C")
        st.write(f"ΔT 24h: {tt.get('dT_24h_C','—')} °C")
        st.write(f"Warm-nose risk: {tt.get('warm_nose_risk','—')}")

    # B) Phase probability
    pp = (obs.get("precip_phase_probability") or {})
    peak_block = pp.get("peak") or {}
    with col2:
        st.markdown("**Precip Phase Probability**")
        if peak_block:
            # simple table
            df_phase = pd.DataFrame([peak_block]).T.reset_index()
            df_phase.columns = ["phase", "prob"]
            st.dataframe(df_phase, use_container_width=True, hide_index=True)

            pmix = compute_p_mix(peak_block)
            st.caption(f"p_mix (sleet + fzra): {pmix:.2f}")
        else:
            st.write("No phase block available.")
            pmix = float("nan")

        if isinstance(pmix, float) and not math.isnan(pmix):
            if pmix >= 0.40:
                st.warning("Mix risk is elevated (sleet/fzra competing with snow).")
            elif pmix >= 0.25:
                st.info("Mix risk is moderate.")
            else:
                st.success("Mix risk is low (phase mostly stable).")

    # C) Timing uncertainty
    tu = (obs.get("timing_uncertainty") or {})
    onset_h = tu.get("onset_window_h")
    peak_h = tu.get("peak_window_h")
    end_h = tu.get("end_window_h")
    with col3:
        st.markdown("**Timing Uncertainty**")
        st.write(f"Onset window: {onset_h if onset_h is not None else '—'} h")
        st.write(f"Peak window: {peak_h if peak_h is not None else '—'} h")
        st.write(f"End window: {end_h if end_h is not None else '—'} h")

    # ---- Triggers ----
    st.markdown("### Coherence Stress Monitor")
    if trig:
        df_trig = pd.DataFrame([{"trigger": k, "status": v} for k, v in trig.items()])
        st.dataframe(df_trig, use_container_width=True, hide_index=True)
    else:
        st.write("No trigger_status provided.")

    # ---- Interpretation ----
    grade, diag = compute_grade_from_update(u)
    trend = confidence_trend(sw)
    arrow = {"up": "▲", "down": "▼", "flat": "→"}.get(trend, "→")
    drop_event = bool(der.get("drop_event", False))

    st.markdown("### Confidence Interpretation")
    st.write(f"**Grade:** {grade} {arrow}  |  **Confidence drop event:** {'YES' if drop_event else 'NO'}")

    reasons = diag.get("reasons", []) or []
    if reasons:
        st.write("Reasons:")
        for r in reasons:
            st.write(f"- {r}")

    st.markdown("### What would make confidence improve?")
    hints = improvement_hints(
        grade=grade,
        pmix=diag.get("p_mix"),
        onset_h=onset_h,
        peak_h=peak_h,
        end_h=end_h,
        trig=trig,
    )
    for h in hints:
        st.write(f"- {h}")

    render_confidence_log(sw)

    #if live_path is not None:
    #    st.caption(f"Live file: {live_path.resolve()}")
    #else:
    #    st.caption("Live file: (none) — running from example (read-only)")



