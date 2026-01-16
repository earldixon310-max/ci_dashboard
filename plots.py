# lib/plots.py
from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from typing import Any, Dict, Optional, Sequence
import numpy as np

def plot_valid_pie(
    dfw: pd.DataFrame,
    title: str = "Valid vs Invalid windows",
    *,
    figsize: tuple[float, float] = (2.8, 2.4),
) -> None:
    if dfw is None or dfw.empty or "valid" not in dfw.columns:
        st.info("No window data available for plotting.")
        return

    valid = int(dfw["valid"].astype(bool).sum())
    invalid = int((~dfw["valid"].astype(bool)).sum())

    fig, ax = plt.subplots(figsize=figsize)
    ax.pie([valid, invalid], labels=["valid", "invalid"], autopct="%1.0f%%")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def plot_delta_hist(
    dfw: pd.DataFrame,
    title: str = "Δ distribution (valid windows)",
    *,
    figsize: tuple[float, float] = (2.8, 2.4),
    bins: int = 10,
) -> None:
    if dfw is None or dfw.empty:
        st.info("No window data available for plotting.")
        return

    d = pd.to_numeric(dfw.loc[dfw["valid"].astype(bool), "delta"], errors="coerce").dropna()
    if d.empty:
        st.info("No valid-window Δ values to plot.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(d.values, bins=bins)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("delta", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def plot_corr_delta_scatter(
    dfw: pd.DataFrame,
    title: str = "|corr| vs Δ (valid windows)",
    *,
    figsize: tuple[float, float] = (2.8, 2.4),
) -> None:
    if dfw is None or dfw.empty:
        st.info("No window data available for plotting.")
        return

    sub = dfw.loc[dfw["valid"].astype(bool)].copy()
    if sub.empty:
        st.info("No valid windows to plot.")
        return

    sub["corr_abs"] = pd.to_numeric(sub["corr"], errors="coerce").abs()
    sub["delta_num"] = pd.to_numeric(sub["delta"], errors="coerce")
    sub = sub.dropna(subset=["corr_abs", "delta_num"])
    if sub.empty:
        st.info("No numeric corr/delta values to plot.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(sub["corr_abs"], sub["delta_num"], s=18)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("|corr|", fontsize=9)
    ax.set_ylabel("delta", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
# lib/plots.py



def plot_corridor_from_report(
    report: Dict[str, Any],
    key: str,
    *,
    title: Optional[str] = None,
) -> None:
    """
    Plot a correlation corridor from report['plots']['corridors'][key].

    Expected schema:
      report['plots']['corridors'][key] = {
        'lags': [..numbers..],
        'corr': [..numbers..],
        'tau_star': number,
        'null_95': number (optional)
      }
    """
    corridors = (((report or {}).get("plots") or {}).get("corridors") or {})
    spec = corridors.get(key)

    if not spec:
        st.info(f"No corridor data found for key: {key}")
        return

    lags = spec.get("lags")
    corr = spec.get("corr")
    tau_star = spec.get("tau_star", None)
    null_95 = spec.get("null_95", None)

    if not isinstance(lags, (list, tuple)) or not isinstance(corr, (list, tuple)):
        st.warning(f"Corridor '{key}' is missing 'lags'/'corr' arrays.")
        return

    if len(lags) == 0 or len(corr) == 0 or len(lags) != len(corr):
        st.warning(f"Corridor '{key}' has invalid array lengths.")
        return

    lags_np = np.asarray(lags, dtype=float)
    corr_np = np.asarray(corr, dtype=float)

    fig, ax = plt.subplots()
    ax.plot(lags_np, corr_np, label="corr(lag)")

    # tau* marker
    if tau_star is not None:
        try:
            ax.axvline(float(tau_star), linestyle="--", label=f"tau* = {tau_star}")
        except Exception:
            pass

    # null 95% bounds
    if null_95 is not None:
        try:
            n95 = float(null_95)
            ax.axhline(+n95, linestyle=":", label="null 95%")
            ax.axhline(-n95, linestyle=":")
        except Exception:
            pass

    ax.set_title(title or f"Corridor: {key}")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Spearman corr")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    st.pyplot(fig, clear_figure=True)

