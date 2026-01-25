# storm_watch_update_from_ci.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from lib.ci_io import load_report, select_bundle
from storm_watch_core import load_json, save_json, ensure_sw_shape, append_update


def _now_iso() -> str:
    # keep it simple + deterministic
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ci_to_storm_update(report: Dict[str, Any], tile_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert CI bundle -> Storm Watch update schema.
    This is the 'real' bridge. You can refine mapping rules over time.
    """
    bundle = select_bundle(report, tile_id=tile_id)
    m = bundle.get("metrics", {}) or {}

    # ---- Your mapping rules (v1) ----
    # Use CI confidence / stability / reliability to drive triggers
    ci_conf = float(m.get("ci_confidence") or 0.0)
    stability = float(m.get("stability") or 0.0)
    reliability = float(m.get("reliability") or 0.0)

    # Example trigger logic (you can tune later)
    trigger_status = {
        "phase_bifurcation_spike": "emerging" if ci_conf < 0.55 else "inactive",
        "timing_window_rewidening": "emerging" if stability < 0.55 else "inactive",
        "band_placement_volatility": "emerging" if reliability < 0.55 else "inactive",
        "accumulation_regime_flip": "inactive",
        "post_storm_hazard_amplification": "inactive",
    }

    # Storm Watch expects some “observations” blocks.
    # If CI doesn’t provide them yet, we leave placeholders and let grade rules handle gracefully.
    update: Dict[str, Any] = {
        "timestamp_iso": _now_iso(),
        "t_minus_h": None,  # optional; fill if you have an event start to compute from
        "observations": {
            "temperature_tendency": {
                "dT_6h_C": None,
                "dT_12h_C": None,
                "dT_24h_C": None,
                "warm_nose_risk": "unknown",
            },
            "timing_uncertainty": {
                "onset_window_h": None,
                "peak_window_h": None,
                "end_window_h": None,
            },
            "precip_phase_probability": {
                "peak": {},  # if you have phase model later, populate here
            },
        },
        "trigger_status": trigger_status,
        "derived": {
            "source": "ci_report",
            "ci_confidence": ci_conf,
            "stability": stability,
            "reliability": reliability,
        },
    }

    return update


def append_from_ci(ci_path: Path, sw_path: Path, tile_id: Optional[str] = None) -> None:
    report = load_report(str(ci_path))
    sw = ensure_sw_shape(load_json(sw_path))
    u = ci_to_storm_update(report, tile_id=tile_id)
    sw = append_update(sw, u)
    save_json(sw_path, sw)


if __name__ == "__main__":
    append_from_ci(Path("ci_report.json"), Path("storm_watch.json"), tile_id=None)
    print("✅ appended update from CI")
