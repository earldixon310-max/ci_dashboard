# storm_watch.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import json
import math
import pandas as pd
import streamlit as st


# -------------------------
# IO
# -------------------------

def load_storm_watch(path: Path | str) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_update(sw: dict) -> Optional[dict]:
    updates = sw.get("updates", []) or []
    if not updates:
        return None
    # Prefer newest by timestamp_iso if present, else by t_minus_h
    def key(u: dict):
        ts = u.get("timestamp_iso")
        tminus = u.get("t_minus_h")
        return (
            pd.to_datetime(ts, errors="coerce") if ts else pd.Timestamp.min,
            float(tminus) if tminus is not None else -1e9,
        )
    return sorted(updates, key=key)[-1]


# -------------------------
# Core math / grading
# -------------------------

def compute_p_mix(block: Dict[str, Any]) -> float:
    """Mix risk proxy: sleet + fzra (defensive)."""
    if not block:
        return float("nan")
    s = float(block.get("sleet", 0.0) or 0.0)
    f = float(block.get("fzra", 0.0) or 0.0)
    return s + f


def max_grade(a: str, b: str) -> str:
    """Return worse grade between a and b (A best, D worst)."""
    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    ra = order.get(a, 2)
    rb = order.get(b, 2)
    return a if ra >= rb else b


def grade_from_rules(
    pmix: float,
    onset_h: Optional[float],
    peak_h: Optional[float],
    end_h: Optional[float],
    trig: Dict[str, str],
    band_stable: Optional[bool] = None,
    regime_stable: Optional[bool] = None,
) -> Tuple[str, List[str]]:
    """
    Return (grade, reasons). A best → D worst.
    Meteorologist framing: “weakest link” (worst violation wins).
    """

    reasons: List[str] = []

    def f(x):
        try:
            return float(x)
        except Exception:
            return None

    onset_h = f(onset_h)
    peak_h = f(peak_h)
    end_h = f(end_h)

    # Trigger flags (active/emerging)
    active = {k for k, v in (trig or {}).items() if str(v).lower() == "active"}
    emerging = {k for k, v in (trig or {}).items() if str(v).lower() == "emerging"}

    # --- Tunable thresholds
    A_pmix, B_pmix, C_pmix = 0.30, 0.40, 0.55
    A_onset, B_onset, C_onset = 4, 6, 10
    A_peak, B_peak, C_peak = 6, 9, 14

    hard_triggers = {"phase_bifurcation_spike", "timing_window_rewidening"}
    soft_triggers = {
        "band_placement_volatility",
        "accumulation_regime_flip",
        "post_storm_hazard_amplification",
    }

    hard_active = len(active.intersection(hard_triggers)) > 0
    soft_active = len(active.intersection(soft_triggers)) > 0

    grade = "A"

    # Mix
    if not (pmix is not None and not (isinstance(pmix, float) and math.isnan(pmix))):
        reasons.append("Mix risk unavailable (no peak phase block).")
        grade = "C"
    elif pmix > C_pmix:
        reasons.append(f"Mix risk very high (p_mix={pmix:.2f} > {C_pmix:.2f}).")
        grade = "D"
    elif pmix > B_pmix:
        reasons.append(f"Mix risk elevated (p_mix={pmix:.2f} > {B_pmix:.2f}).")
        grade = max_grade(grade, "C")
    elif pmix > A_pmix:
        reasons.append(f"Mix risk moderate (p_mix={pmix:.2f} > {A_pmix:.2f}).")
        grade = max_grade(grade, "B")

    # Timing
    if onset_h is not None:
        if onset_h > C_onset:
            reasons.append(f"Onset window wide ({onset_h:.0f}h > {C_onset}h).")
            grade = max_grade(grade, "D")
        elif onset_h > B_onset:
            reasons.append(f"Onset window still wide ({onset_h:.0f}h > {B_onset}h).")
            grade = max_grade(grade, "C")
        elif onset_h > A_onset:
            reasons.append(f"Onset window moderate ({onset_h:.0f}h > {A_onset}h).")
            grade = max_grade(grade, "B")

    if peak_h is not None:
        if peak_h > C_peak:
            reasons.append(f"Peak window wide ({peak_h:.0f}h > {C_peak}h).")
            grade = max_grade(grade, "D")
        elif peak_h > B_peak:
            reasons.append(f"Peak window still wide ({peak_h:.0f}h > {B_peak}h).")
            grade = max_grade(grade, "C")
        elif peak_h > A_peak:
            reasons.append(f"Peak window moderate ({peak_h:.0f}h > {A_peak}h).")
            grade = max_grade(grade, "B")

    # Triggers
    if hard_active:
        reasons.append("Hard stress trigger active (phase bifurcation or timing re-widening).")
        grade = max_grade(grade, "C")
    if soft_active:
        reasons.append("Soft stress trigger active (band/regime/hazard volatility).")
        grade = max_grade(grade, "C")
    if len(active) >= 3:
        reasons.append("Multiple triggers active (system not settled).")
        grade = max_grade(grade, "D")

    # Optional stability flags
    if band_stable is False:
        reasons.append("Band placement not stable across guidance cycles.")
        grade = max_grade(grade, "C")
    if regime_stable is False:
        reasons.append("Accumulation regime not stable.")
        grade = max_grade(grade, "C")

    if not reasons:
        reasons.append("All confidence gates passed (mix + timing stable).")

    return grade, reasons


def compute_grade_from_update(u: dict) -> Tuple[str, dict]:
    """Returns (grade, diagnostics) computed from the update (rule-driven)."""
    obs = u.get("observations", {}) or {}
    trig = u.get("trigger_status", {}) or {}
    tu = (obs.get("timing_uncertainty") or {}) or {}
    pp = (obs.get("precip_phase_probability") or {}) or {}
    peak = pp.get("peak") or {}

    pmix = compute_p_mix(peak) if peak else float("nan")
    onset_h = tu.get("onset_window_h")
    peak_h = tu.get("peak_window_h")
    end_h = tu.get("end_window_h")

    grade, reasons = grade_from_rules(
        pmix=pmix,
        onset_h=onset_h,
        peak_h=peak_h,
        end_h=end_h,
        trig=trig,
    )

    diag = {
        "p_mix": None if (isinstance(pmix, float) and math.isnan(pmix)) else float(pmix),
        "onset_window_h": onset_h,
        "peak_window_h": peak_h,
        "end_window_h": end_h,
        "reasons": reasons,
    }
    return grade, diag


def improvement_hints(
    grade: str,
    pmix: float,
    onset_h: Optional[float],
    peak_h: Optional[float],
    end_h: Optional[float],
    trig: Dict[str, str],
) -> List[str]:
    """Generate 'what would improve confidence' hints based on current values."""
    hints: List[str] = []

    def f(x):
        try:
            return float(x)
        except Exception:
            return None

    onset_h = f(onset_h)
    peak_h = f(peak_h)
    end_h = f(end_h)

    if pmix is None or (isinstance(pmix, float) and math.isnan(pmix)):
        hints.append("Add/confirm a peak precip-type probability block so mix risk can be evaluated.")
    else:
        if pmix >= 0.40:
            hints.append("Mix risk should trend down: p_mix < 0.35 supports a cleaner precip-type outcome.")
        elif pmix >= 0.30:
            hints.append("Mix risk should trend down further: p_mix < 0.30 supports Grade A precip type.")

    if onset_h is not None and onset_h > 6:
        hints.append("Onset timing needs to tighten: onset window ≤ 6h would raise confidence.")
    elif onset_h is not None and onset_h > 4:
        hints.append("Onset timing needs to tighten: onset window ≤ 4h supports Grade A.")

    if peak_h is not None and peak_h > 9:
        hints.append("Peak timing needs to tighten: peak window ≤ 9h would raise confidence.")
    elif peak_h is not None and peak_h > 6:
        hints.append("Peak timing needs to tighten: peak window ≤ 6h supports Grade A.")

    if end_h is not None and end_h > 10:
        hints.append("End timing uncertainty is still broad — a narrower end window reduces tail-risk uncertainty.")

    t = {k: str(v).lower() for k, v in (trig or {}).items()}
    if t.get("phase_bifurcation_spike") in ("active", "emerging"):
        hints.append("Phase bifurcation needs to settle: reduced sleet/fzra competition across updates.")
    if t.get("timing_window_rewidening") in ("active", "emerging"):
        hints.append("Timing re-widening must stop: successive updates should show shrinking onset/peak windows.")
    if t.get("band_placement_volatility") in ("active", "emerging"):
        hints.append("Band placement must stabilize: consistent axis/placement across guidance cycles raises confidence.")
    if t.get("accumulation_regime_flip") in ("active", "emerging"):
        hints.append("Regime flip must stabilize: guidance should converge on a single accumulation mechanism.")
    if t.get("post_storm_hazard_amplification") in ("active", "emerging"):
        hints.append("Post-storm hazard signal should clarify: confirm refreeze/flash-freeze with temps + wind.")

    if not hints and grade in ("A", "B"):
        hints.append("Maintain consistency: continued agreement across updates should keep or improve the grade.")

    return hints


def confidence_trend(sw: dict) -> str:
    """Up/down/flat based on last two grades."""
    updates = sw.get("updates", []) or []
    if len(updates) < 2:
        return "flat"
    gmap = {"A": 4, "B": 3, "C": 2, "D": 1}
    g1, _ = compute_grade_from_update(updates[-2])
    g2, _ = compute_grade_from_update(updates[-1])
    v1, v2 = gmap.get(g1, 2), gmap.get(g2, 2)
    if v2 > v1:
        return "up"
    if v2 < v1:
        return "down"
    return "flat"


# -------------------------
# Storm clock (this fixes your “missing info”)
# -------------------------

CHECKPOINTS = [-72, -48, -36, -24, -18, -12, -6, -3, 0]

def _pick_update_for_checkpoint(updates: List[dict], checkpoint: int) -> Optional[dict]:
    """
    Choose the update closest to this checkpoint.
    Example: if you only have -33 and -27, the -36 row will snap to -33.
    """
    if not updates:
        return None

    def dist(u: dict):
        t = u.get("t_minus_h")
        if t is None:
            return 1e9
        return abs(float(t) - float(checkpoint))

    return sorted(updates, key=dist)[0]


def checkpoint_rows(sw: dict) -> List[dict]:
    updates = sw.get("updates", []) or []
    rows: List[dict] = []

    for cp in CHECKPOINTS:
        u = _pick_update_for_checkpoint(updates, cp)
        if u is None:
            rows.append({"T-hrs": cp, "time": "—", "grade": "—", "drop": ""})
            continue

        grade, _diag = compute_grade_from_update(u)
        drop_event = bool((u.get("derived") or {}).get("drop_event", False))
        ts = u.get("timestamp_iso") or "—"

        rows.append({
            "T-hrs": cp,
            "time": ts,
            "grade": grade,
            "drop": "●" if drop_event else "",
        })

    return rows


# -------------------------
# Charts (Altair)
# -------------------------

def phase_bar(block: dict, title: str = "Peak phase mix"):
    import altair as alt

    phases = ["snow", "sleet", "fzra", "rain"]
    rows = [{"phase": p, "prob": float(block.get(p, 0.0) or 0.0)} for p in phases]
    df = pd.DataFrame(rows)

    s = df["prob"].sum()
    if s > 0:
        df["prob"] = df["prob"] / s

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("prob:Q", stack="normalize", title=None),
            y=alt.Y("label:N", title=None),
            color=alt.Color("phase:N", title=None),
            tooltip=["phase:N", alt.Tooltip("prob:Q", format=".2f")],
        )
        .transform_calculate(label=f'"{title}"')
        .properties(height=55)
    )
    st.altair_chart(chart, use_container_width=True)


def confidence_strip(sw: dict, title: str = "Confidence trajectory (grades over time)"):
    import altair as alt

    updates = sw.get("updates", []) or []
    if not updates:
        st.info("No updates available for confidence strip.")
        return

    gmap = {"A": 4, "B": 3, "C": 2, "D": 1}

    rows = []
    for u in updates:
        grade, diag = compute_grade_from_update(u)
        ts = u.get("timestamp_iso")
        tminus = u.get("t_minus_h")

        rows.append({
            "timestamp_iso": ts,
            "t_minus_h": float(tminus) if tminus is not None else None,
            "grade": grade,
            "score": gmap.get(grade, None),
            "p_mix": float(diag.get("p_mix") or 0.0),
            "peak_window_h": float(diag.get("peak_window_h") or 0.0),
            "onset_window_h": float(diag.get("onset_window_h") or 0.0),
        })

    df = pd.DataFrame(rows)
    use_time = df["timestamp_iso"].notna().all()

    if use_time:
        df["timestamp_iso"] = pd.to_datetime(df["timestamp_iso"], errors="coerce")
        df = df.sort_values("timestamp_iso")
        x_enc = alt.X("timestamp_iso:T", title=None)
    else:
        df = df.sort_values("t_minus_h")
        x_enc = alt.X("t_minus_h:Q", title="T-hrs")

    base = alt.Chart(df).properties(height=90)

    line = base.mark_line(point=False).encode(
        x=x_enc,
        y=alt.Y("score:Q", title=None, scale=alt.Scale(domain=[1, 4])),
        tooltip=[
            alt.Tooltip("grade:N", title="Grade"),
            alt.Tooltip("p_mix:Q", format=".2f", title="p_mix"),
            alt.Tooltip("peak_window_h:Q", format=".0f", title="Peak win (h)"),
            alt.Tooltip("onset_window_h:Q", format=".0f", title="Onset win (h)"),
            (alt.Tooltip("timestamp_iso:T", title="Time") if use_time else alt.Tooltip("t_minus_h:Q", title="T-hrs")),
        ],
    )

    points = base.mark_point(size=90).encode(x=x_enc, y=alt.Y("score:Q", title=None))
    labels = base.mark_text(dy=-12).encode(x=x_enc, y="score:Q", text="grade:N")

    st.markdown(f"### {title}")
    st.altair_chart(line + points + labels, use_container_width=True)
    st.caption("Grade scale: A (high confidence) → D (low confidence). Hover a point to see what drove the grade.")


# -------------------------
# Delta narrative for the log
# -------------------------

def safe_get_pmix(u: Dict) -> Optional[float]:
    obs = (u.get("observations", {}) or {})
    pp = (obs.get("precip_phase_probability") or {})
    peak_block = pp.get("peak") or {}
    if not peak_block:
        return float("nan")
    return compute_p_mix(peak_block)


def what_changed(prev_u: Dict, u: Dict) -> str:
    if not prev_u:
        return "First logged update."

    pprev = safe_get_pmix(prev_u)
    pnow = safe_get_pmix(u)

    prev_tu = (prev_u.get("observations", {}) or {}).get("timing_uncertainty", {}) or {}
    now_tu = (u.get("observations", {}) or {}).get("timing_uncertainty", {}) or {}

    prev_peak = prev_tu.get("peak_window_h")
    now_peak = now_tu.get("peak_window_h")

    parts: List[str] = []

    if pprev is not None and pnow is not None and not any(map(math.isnan, [pprev, pnow])):
        if pnow > pprev + 0.02:
            parts.append("Mix probability increased in peak block")
        elif pnow < pprev - 0.02:
            parts.append("Mix probability decreased in peak block")
        else:
            parts.append("Mix probability roughly steady")

    try:
        if prev_peak is not None and now_peak is not None:
            if float(now_peak) > float(prev_peak) + 0.5:
                parts.append("Peak window widened")
            elif float(now_peak) < float(prev_peak) - 0.5:
                parts.append("Peak window tightened")
            else:
                parts.append("Peak window roughly steady")
    except Exception:
        pass

    return "; ".join(parts) if parts else "Update captured (no deltas computed)."


# -------------------------
# Main renderer (Storm Watch page)
# -------------------------

def render_storm_watch(sw: dict):
    st.title("Storm Watch Mode")
    st.caption("Operator view — confidence-first storm tracking (no deterministic totals).")

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

    # Storm Clock (checkpoint-filled)
    st.markdown("### Storm Clock (T-72 → T-0)")
    rows = checkpoint_rows(sw)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Confidence strip
    confidence_strip(sw, title="Confidence trajectory (grades over time)")

    # Current snapshot
    u = latest_update(sw)
    if not u:
        st.warning("No latest update available.")
        return

    obs = u.get("observations", {}) or {}
    trig = u.get("trigger_status", {}) or {}
    der = u.get("derived", {}) or {}

    st.markdown("### Current Snapshot")
    col1, col2, col3 = st.columns([1.0, 1.6, 0.9], gap="large")

    # A) Temperature
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

        pmix = float("nan")
        if peak_block:
            pmix = compute_p_mix(peak_block)

            phase_bar(peak_block, title="Peak phase mix")

            df_phase = pd.DataFrame([peak_block]).T.reset_index()
            df_phase.columns = ["phase", "prob"]
            st.dataframe(df_phase, use_container_width=True, hide_index=True)

            st.caption(f"p_mix (sleet + fzra): {pmix:.2f}")
        else:
            st.write("No phase block available.")

        if isinstance(pmix, float) and not math.isnan(pmix):
            if pmix >= 0.40:
                st.warning("Mix risk is elevated (sleet/fzra competing with snow).")
            elif pmix >= 0.25:
                st.info("Mix risk is moderate.")
            else:
                st.success("Mix risk is low (phase mostly stable).")

    # C) Timing
    tu = (obs.get("timing_uncertainty") or {})
    onset_h = tu.get("onset_window_h")
    peak_h = tu.get("peak_window_h")
    end_h = tu.get("end_window_h")

    with col3:
        st.markdown("**Timing Uncertainty**")
        st.write(f"Onset window: {onset_h if onset_h is not None else '—'} h")
        st.write(f"Peak window: {peak_h if peak_h is not None else '—'} h")
        st.write(f"End window: {end_h if end_h is not None else '—'} h")

    # Grade + reasons + hints
    grade_calc, reasons = grade_from_rules(
        pmix=compute_p_mix(peak_block) if peak_block else float("nan"),
        onset_h=onset_h,
        peak_h=peak_h,
        end_h=end_h,
        trig=trig,
    )
    drop_event = bool(der.get("drop_event", False))
    trend = confidence_trend(sw)
    arrow = {"up": "▲", "down": "▼", "flat": "→"}.get(trend, "→")

    st.markdown("### Coherence Stress Monitor")
    if trig:
        df_trig = pd.DataFrame([{"trigger": k, "status": v} for k, v in trig.items()])
        st.dataframe(df_trig, use_container_width=True, hide_index=True)
    else:
        st.write("No trigger_status provided.")

    st.markdown("### Confidence Interpretation")
    st.write(f"**Grade:** {grade_calc} {arrow}  |  **Confidence drop event:** {'YES' if drop_event else 'NO'}")

    if reasons:
        st.write("Reasons:")
        for r in reasons:
            st.write(f"- {r}")

    st.markdown("### What would make confidence improve?")
    pmix_now = compute_p_mix(peak_block) if peak_block else float("nan")
    hints = improvement_hints(grade_calc, pmix_now, onset_h, peak_h, end_h, trig)
    for h in hints:
        st.write(f"- {h}")

    # Log
    st.markdown("### Confidence Log (All Updates)")
    updates = sw.get("updates", []) or []

    if updates:
        rows = []
        prev = None
        for urow in updates:
            obs_i = (urow.get("observations", {}) or {})
            trig_i = (urow.get("trigger_status", {}) or {})
            tu_i = (obs_i.get("timing_uncertainty") or {}) or {}
            pp_i = (obs_i.get("precip_phase_probability") or {}) or {}
            peak_i = pp_i.get("peak") or {}
            pmix_i = compute_p_mix(peak_i) if peak_i else float("nan")

            g_i, _reasons_i = grade_from_rules(
                pmix=pmix_i,
                onset_h=tu_i.get("onset_window_h"),
                peak_h=tu_i.get("peak_window_h"),
                end_h=tu_i.get("end_window_h"),
                trig=trig_i,
            )

            rows.append({
                "Time": urow.get("timestamp_iso"),
                "T-hrs": urow.get("t_minus_h"),
                "Grade": g_i,
                "Mix risk (p_mix)": None if math.isnan(pmix_i) else round(pmix_i, 2),
                "Peak window (h)": tu_i.get("peak_window_h"),
                "What changed": what_changed(prev, urow),
            })
            prev = urow

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.write("No update history.")
