# lib/ci_math.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import pandas as pd

# --- Tier thresholds: policy (NOT operator-tuned) ---
TH_ADMISSIBLE = 0.85
TH_CONDITIONAL = 0.65

def tier_from_ci(ci_conf: float) -> Tuple[str, str]:
    if ci_conf >= TH_ADMISSIBLE:
        return ("Admissible", "admissible")
    if ci_conf >= TH_CONDITIONAL:
        return ("Conditional", "conditional")
    return ("Exploratory", "exploratory")

def clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v != v:
        return 0.0
    return max(0.0, min(1.0, v))

def windows_to_df(windows: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for w in windows or []:
        rows.append({
            "index": w.get("index"),
            "start": w.get("start"),
            "end": w.get("end"),
            "valid": bool(w.get("valid")),
            "tau_star": w.get("tau_star"),
            "corr": w.get("corr"),
            "delta": w.get("delta"),
            "validity_reason": w.get("validity_reason"),
            "tile_id": w.get("tile_id"),
        })
    return pd.DataFrame(rows)

def compute_conf_from_schema(dfw: pd.DataFrame) -> pd.DataFrame:
    """
    Window confidence is a *viewer aid*, not the CI gating score.
    For valid windows: combine |corr| and delta.
    For invalid windows: confidence = 0.
    """
    if dfw is None or dfw.empty:
        return dfw

    df = dfw.copy()

    # default zero if invalid
    df["confidence"] = 0.0

    valid_mask = df["valid"].astype(bool)
    if valid_mask.any():
        corr = pd.to_numeric(df.loc[valid_mask, "corr"], errors="coerce").fillna(0.0).abs()
        delta = pd.to_numeric(df.loc[valid_mask, "delta"], errors="coerce").fillna(0.0)

        # map: corr <0.2 weak, >=0.8 strong
        s_corr = ((corr - 0.2) / 0.6).clip(lower=0.0, upper=1.0)
        # map delta (0..0.35) into 0..1
        s_delta = (delta / 0.35).clip(lower=0.0, upper=1.0)

        C = (0.55 * s_corr + 0.45 * s_delta).clip(lower=0.0, upper=1.0)
        df.loc[valid_mask, "confidence"] = C

    def label(c: float) -> str:
        if c >= 0.85: return "VERY HIGH"
        if c >= 0.70: return "HIGH"
        if c >= 0.50: return "MODERATE"
        return "LOW"

    df["confidence_label"] = df["confidence"].apply(lambda x: label(float(x)))
    return df

def conf_summary(df_conf: pd.DataFrame) -> Dict[str, Any]:
    if df_conf is None or df_conf.empty or "confidence" not in df_conf.columns:
        return {"n": 0, "confidence_median": None, "high_frac": None}
    med = float(df_conf["confidence"].median())
    high = float((df_conf["confidence"] >= 0.70).mean())
    return {"n": int(len(df_conf)), "confidence_median": med, "high_frac": high}
