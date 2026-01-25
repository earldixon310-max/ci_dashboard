# storm_watch_core.py
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# -------------------------
# IO
# -------------------------

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def now_iso_local() -> str:
    # Keep it simple: local wall time ISO, no tz conversion assumptions.
    return datetime.now().astimezone().isoformat(timespec="seconds")


# -------------------------
# Storm Watch schema helpers
# -------------------------

def ensure_sw_shape(sw: Dict[str, Any]) -> Dict[str, Any]:
    sw = copy.deepcopy(sw or {})
    sw.setdefault("target", {})
    sw.setdefault("event", {})
    sw.setdefault("updates", [])
    if not isinstance(sw["updates"], list):
        sw["updates"] = []
    return sw

def latest_update(sw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ups = sw.get("updates", []) or []
    if not ups:
        return None

    # Prefer timestamp ordering if possible
    def key(u: Dict[str, Any]) -> Tuple[int, str]:
        ts = str(u.get("timestamp_iso") or "")
        tminus = u.get("t_minus_h")
        try:
            tminus_k = int(float(tminus))
        except Exception:
            tminus_k = 10**9
        return (0 if ts else 1, ts)  # ts first

    ups_sorted = sorted(ups, key=key)
    return ups_sorted[-1]


# -------------------------
# Confidence rules (same spirit as what you already wrote)
# -------------------------

def compute_p_mix(peak_block: Dict[str, Any]) -> float:
    if not peak_block:
        return float("nan")
    s = float(peak_block.get("sleet", 0.0) or 0.0)
    f = float(peak_block.get("fzra", 0.0) or 0.0)
    return s + f

def max_grade(a: str, b: str) -> str:
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
) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    def f(x):
        try:
            return float(x)
        except Exception:
            return None

    onset_h = f(onset_h)
    peak_h = f(peak_h)
    end_h = f(end_h)

    active = {k for k, v in (trig or {}).items() if str(v).lower() == "active"}

    # Tunables
    A_pmix, B_pmix, C_pmix = 0.30, 0.40, 0.55
    A_onset, B_onset, C_onset = 4, 6, 10
    A_peak, B_peak, C_peak = 6, 9, 14

    hard_triggers = {"phase_bifurcation_spike", "timing_window_rewidening"}
    soft_triggers = {"band_placement_volatility", "accumulation_regime_flip", "post_storm_hazard_amplification"}

    hard_active = len(active.intersection(hard_triggers)) > 0
    soft_active = len(active.intersection(soft_triggers)) > 0

    grade = "A"

    # Mix
    if not (pmix is not None and isinstance(pmix, float) and not math.isnan(pmix)):
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

    if not reasons:
        reasons.append("All confidence gates passed (mix + timing stable).")

    return grade, reasons


def compute_grade_from_update(u: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    obs = (u.get("observations", {}) or {})
    trig = (u.get("trigger_status", {}) or {})
    tu = (obs.get("timing_uncertainty", {}) or {})
    pp = (obs.get("precip_phase_probability", {}) or {})
    peak = (pp.get("peak") or {})
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
        "p_mix": pmix,
        "onset_window_h": onset_h,
        "peak_window_h": peak_h,
        "end_window_h": end_h,
        "reasons": reasons,
    }
    return grade, diag


def confidence_trend(sw: Dict[str, Any]) -> str:
    """Compare last two grades: up/down/flat."""
    ups = sw.get("updates", []) or []
    if len(ups) < 2:
        return "flat"

    gmap = {"A": 4, "B": 3, "C": 2, "D": 1}

    g1, _ = compute_grade_from_update(ups[-2])
    g2, _ = compute_grade_from_update(ups[-1])
    s1 = gmap.get(g1)
    s2 = gmap.get(g2)
    if s1 is None or s2 is None:
        return "flat"
    if s2 > s1:
        return "up"
    if s2 < s1:
        return "down"
    return "flat"


def improvement_hints(
    grade: str,
    pmix: float,
    onset_h: Optional[float],
    peak_h: Optional[float],
    end_h: Optional[float],
    trig: Dict[str, str],
) -> List[str]:
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
            hints.append("Mix risk should trend down further: p_mix < 0.30 supports high-confidence precip type.")

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


# -------------------------
# Storm clock rows (T-72 → T-0) from latest known update per checkpoint
# -------------------------

def checkpoint_rows(sw: Dict[str, Any], checkpoints: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    checkpoints = checkpoints or [-72, -48, -36, -24, -18, -12, -6, -3, 0]
    updates = sw.get("updates", []) or []
    if not updates:
        return []

    # Map checkpoint -> nearest update (closest t_minus_h)
    def get_tminus(u) -> Optional[float]:
        try:
            return float(u.get("t_minus_h"))
        except Exception:
            return None

    rows: List[Dict[str, Any]] = []
    for ck in checkpoints:
        best_u = None
        best_dist = 1e18
        for u in updates:
            tm = get_tminus(u)
            if tm is None:
                continue
            d = abs(tm - ck)
            if d < best_dist:
                best_dist = d
                best_u = u

        if best_u is None:
            rows.append({"T-hrs": ck, "time": "—", "grade": "—", "drop": ""})
        else:
            g, diag = compute_grade_from_update(best_u)
            drop = bool((best_u.get("derived") or {}).get("drop_event", False))
            rows.append({
                "T-hrs": ck,
                "time": best_u.get("timestamp_iso") or "—",
                "grade": g,
                "drop": drop,
            })
    return rows


# -------------------------
# CI → Storm Watch conversion
# -------------------------

def ci_report_to_storm_update(
    ci_report_path: Path,
    tile_id: Optional[str] = None,
    timestamp_iso: Optional[str] = None,
    t_minus_h: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Converts CI outputs to a Storm Watch update.
    This is intentionally heuristic: CI doesn't contain met fields like precip phase, so those are left empty.
    """
    from lib.ci_io import load_report, available_tiles, select_bundle
    from lib.ci_math import windows_to_df, compute_conf_from_schema

    report = load_report(str(ci_report_path))
    bundle = select_bundle(report, tile_id=tile_id)

    dfw = windows_to_df(bundle.get("windows", []) or [])
    dfc = compute_conf_from_schema(dfw)

    # --- timing_uncertainty from spread of window start/end (hours)
    # Use valid windows if possible
    if dfc is not None and not dfc.empty:
        d = dfc.copy()
        if "valid" in d.columns:
            d = d[d["valid"] == True]  # noqa: E712
        # if nothing valid, fall back to all
        if d.empty:
            d = dfc.copy()

        # Try to compute hour spans using start/end if they exist
        onset_h = peak_h = end_h = None

        try:
            if "start" in d.columns and "end" in d.columns:
                stt = pd.to_datetime(d["start"], errors="coerce")
                enn = pd.to_datetime(d["end"], errors="coerce")
                stt = stt.dropna()
                enn = enn.dropna()
                if len(stt) >= 3:
                    onset_h = float((stt.quantile(0.9) - stt.quantile(0.1)).total_seconds() / 3600.0)
                if len(enn) >= 3:
                    end_h = float((enn.quantile(0.9) - enn.quantile(0.1)).total_seconds() / 3600.0)
        except Exception:
            pass

        # peak window proxy: spread of tau_star (if available) mapped to hours
        try:
            if "tau_star" in d.columns:
                ts = pd.to_numeric(d["tau_star"], errors="coerce").dropna()
                if len(ts) >= 3:
                    # tau_star is hours already in your CI pipeline
                    peak_h = float(ts.quantile(0.9) - ts.quantile(0.1))
        except Exception:
            pass
    else:
        onset_h = peak_h = end_h = None

    m = (bundle.get("metrics", {}) or {})
    ci_conf = float(m.get("ci_confidence") or 0.0)
    stability = float(m.get("stability") or 0.0)
    reliability = float(m.get("reliability") or 0.0)

    # --- triggers from CI metrics (simple but effective)
    # You can tune these thresholds as you gather data.
    trig: Dict[str, str] = {
        "phase_bifurcation_spike": "inactive",          # CI doesn't know phase; left inactive
        "timing_window_rewidening": "inactive",
        "band_placement_volatility": "inactive",
        "accumulation_regime_flip": "inactive",         # CI doesn't know regime; left inactive
        "post_storm_hazard_amplification": "inactive",   # CI doesn't know hazard; left inactive
    }

    if reliability < 0.35:
        trig["timing_window_rewidening"] = "active"
    elif reliability < 0.55:
        trig["timing_window_rewidening"] = "emerging"

    if stability < 0.35:
        trig["band_placement_volatility"] = "active"
    elif stability < 0.55:
        trig["band_placement_volatility"] = "emerging"

    # CI-confidence itself can act like a “system settling” meta-trigger
    if ci_conf < 0.25:
        trig["post_storm_hazard_amplification"] = "emerging"

    update: Dict[str, Any] = {
        "timestamp_iso": timestamp_iso or now_iso_local(),
        "t_minus_h": t_minus_h,
        "source": {
            "type": "ci_report",
            "tile_id": tile_id or "GLOBAL",
            "ci_confidence": ci_conf,
            "stability": stability,
            "reliability": reliability,
        },
        "observations": {
            "temperature_tendency": {},  # CI doesn't provide this
            "precip_phase_probability": {},  # CI doesn't provide this
            "timing_uncertainty": {
                "onset_window_h": onset_h,
                "peak_window_h": peak_h,
                "end_window_h": end_h,
            },
        },
        "trigger_status": trig,
        "derived": {
            "drop_event": False,  # will be recomputed/overridden if you want later
        },
    }

    # Attach a computed grade (rule-driven, but will likely be "C" because pmix is NaN)
    g, diag = compute_grade_from_update(update)
    update["derived"]["confidence_grade"] = g
    update["derived"]["grade_reasons"] = diag.get("reasons", [])

    return update


# -------------------------
# Simulate next update (demo button + CLI)
# -------------------------

def simulate_next_update(sw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a plausible next update by nudging windows + triggers a bit.
    Does NOT require CI. Good for demos.
    """
    sw = ensure_sw_shape(sw)
    last = latest_update(sw) or {}

    # Start from last, then mutate gently
    u = copy.deepcopy(last) if last else {"observations": {}, "trigger_status": {}, "derived": {}}
    u.setdefault("observations", {})
    u.setdefault("trigger_status", {})
    u.setdefault("derived", {})
    u["timestamp_iso"] = now_iso_local()

    # t_minus: move toward 0 if present
    try:
        tm = float(u.get("t_minus_h"))
        # move forward by ~6 hours
        u["t_minus_h"] = tm + 6.0
    except Exception:
        pass

    # timing_uncertainty: tighten slightly (typical convergence)
    tu = (u["observations"].get("timing_uncertainty") or {})
    for k, shrink in [("onset_window_h", 0.8), ("peak_window_h", 0.85), ("end_window_h", 0.9)]:
        try:
            v = float(tu.get(k))
            tu[k] = max(1.0, v * shrink)
        except Exception:
            pass
    u["observations"]["timing_uncertainty"] = tu

    # phase mix: if present, reduce sleet/fzra slightly
    pp = (u["observations"].get("precip_phase_probability") or {})
    peak = (pp.get("peak") or {})
    if peak:
        for k in ("sleet", "fzra"):
            try:
                peak[k] = max(0.0, float(peak.get(k, 0.0)) * 0.85)
            except Exception:
                pass
        # Renormalize
        s = sum(float(peak.get(x, 0.0) or 0.0) for x in ("snow", "sleet", "fzra", "rain"))
        if s > 0:
            for k in ("snow", "sleet", "fzra", "rain"):
                peak[k] = float(peak.get(k, 0.0) or 0.0) / s
        pp["peak"] = peak
        u["observations"]["precip_phase_probability"] = pp

    # triggers: decay "emerging" -> "inactive", "active" -> "emerging"
    trig = {k: str(v).lower() for k, v in (u.get("trigger_status") or {}).items()}
    new_trig: Dict[str, str] = {}
    for k, v in trig.items():
        if v == "active":
            new_trig[k] = "emerging"
        elif v == "emerging":
            new_trig[k] = "inactive"
        else:
            new_trig[k] = "inactive"
    u["trigger_status"] = new_trig

    # derived recompute
    g, diag = compute_grade_from_update(u)
    u["derived"]["confidence_grade"] = g
    u["derived"]["grade_reasons"] = diag.get("reasons", [])
    u["derived"]["drop_event"] = False

    return u


# -------------------------
# Append + recompute
# -------------------------

def append_update(sw: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    sw = ensure_sw_shape(sw)
    sw["updates"].append(update)
    return sw
