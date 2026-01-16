# lib/ci_ui.py
from __future__ import annotations
import streamlit as st

TEAL = "#18A7A7"
TEAL_DARK = "#0F7E7E"
CANVAS = "#F2F4F7"
CARD = "#FFFFFF"
BORDER = "rgba(16, 24, 40, 0.10)"
TEXT = "#1E2A32"
MUTED = "rgba(30, 42, 50, 0.70)"

TIER_COLORS = {
    "exploratory": {"strip": "#B42318", "bg": "rgba(180, 35, 24, 0.10)"},
    "conditional": {"strip": "#F79009", "bg": "rgba(247, 144, 9, 0.12)"},
    "admissible": {"strip": "#12B76A", "bg": "rgba(18, 183, 106, 0.12)"},
    "unknown": {"strip": "rgba(16,24,40,0.35)", "bg": "rgba(16,24,40,0.04)"},
}


def inject_reference_css(app_title: str = "CI Dashboard") -> None:
    st.markdown(
        f"""
<style>
/* --- Global canvas --- */
html, body, [class*="css"] {{
  background: {CANVAS} !important;
  color: {TEXT};
}}

/* Remove default Streamlit padding a bit */
.block-container {{
  padding-top: 0.8rem;
  padding-bottom: 1.2rem;
}}

/* Hide Streamlit default menu/footer */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

/* --- Top nav bar imitation --- */
.ci-topbar {{
  width: 100%;
  background: linear-gradient(90deg, {TEAL_DARK}, {TEAL});
  border-radius: 10px;
  padding: 14px 16px;
  color: white;
  box-shadow: 0 10px 24px rgba(0,0,0,0.12);
  margin-bottom: 12px;
  display:flex;
  align-items:center;
  justify-content:space-between;
}}
.ci-topbar .title {{
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0.2px;
}}
.ci-topbar .meta {{
  font-size: 12px;
  opacity: 0.88;
}}

/* --- Card styles --- */
.ci-card {{
  background: {CARD};
  border: 1px solid {BORDER};
  border-radius: 10px;
  padding: 14px 14px;
  box-shadow: 0 8px 20px rgba(16,24,40,0.08);
}}
.ci-card h3 {{
  margin: 0 0 8px 0;
  font-size: 16px;
}}
.small-muted {{
  color: {MUTED};
  font-size: 12px;
}}

/* --- Tier status card (colored strip) --- */
.tier-card {{
  display:flex;
  gap:0;
  background: transparent;
  border-radius: 10px;
  overflow:hidden;
  border: 1px solid {BORDER};
  box-shadow: 0 8px 20px rgba(16,24,40,0.08);
  margin-bottom: 10px;
}}
.tier-strip {{ width: 10px; }}
.tier-body {{
  width: 100%;
  padding: 14px 14px;
  background: {CARD};
}}


/* --- Metric tiles like appliance dashboards --- */
.metric-grid {{
  display:grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-top: 10px;
}}
.metric-tile {{
  background: {CARD};
  border: 1px solid {BORDER};
  border-radius: 10px;
  padding: 12px 12px;
  box-shadow: 0 6px 14px rgba(16,24,40,0.06);
}}
.metric-label {{
  color: {MUTED};
  font-size: 12px;
}}
.metric-value {{
  font-size: 28px;
  font-weight: 700;
  margin-top: 6px;
  line-height: 1.0;
}}

/* --- Tables: clean and compact --- */
div[data-testid="stDataFrame"] {{
  background: {CARD};
  border: 1px solid {BORDER};
  border-radius: 10px;
  padding: 6px;
  box-shadow: 0 6px 14px rgba(16,24,40,0.06);
}}

/* --- Section separators --- */
hr.sep {{
  border: none;
  height: 1px;
  background: rgba(16,24,40,0.10);
  margin: 16px 0;
}}
</style>

<div class="ci-topbar">
  <div class="title">{app_title}</div>
  <div class="meta">Operator view • logic center</div>
</div>
""",
        unsafe_allow_html=True,
    )



def render_ci_summary_block(
    *,
    tier_name: str,
    tier_code: str,
    ci_conf: float,
    presence: float,
    stability: float,
    reliability: float,
    explanation: str = "",
) -> None:
    ICON = {
        "exploratory": "🔴",
        "conditional": "🟡",
        "admissible": "🟢",
    }.get(tier_code, "⚪")

    tier_color = TIER_COLORS.get(tier_code, TIER_COLORS["unknown"])
    strip = tier_color["strip"]

    html = f"""
<div class="tier-card">
  <div class="tier-strip" style="background:{strip};"></div>
  <div class="tier-body">
    <h3 style="margin:0 0 8px 0; font-size:16px;">Status</h3>

    <div style="font-weight:700; font-size:1.05rem; margin-bottom:10px;">
      {ICON} {tier_name} (CI tier)
    </div>

    <div class="metric-grid">
      <div class="metric-tile"><div class="metric-label">CI Confidence</div><div class="metric-value">{ci_conf:.3f}</div></div>
      <div class="metric-tile"><div class="metric-label">Presence</div><div class="metric-value">{presence:.3f}</div></div>
      <div class="metric-tile"><div class="metric-label">Stability</div><div class="metric-value">{stability:.3f}</div></div>
      <div class="metric-tile"><div class="metric-label">Reliability</div><div class="metric-value">{reliability:.3f}</div></div>
    </div>

    <div class="small-muted" style="margin-top:10px;">
      CI Confidence {ci_conf:.3f} — presence {presence:.3f}, stability {stability:.3f}, reliability {reliability:.3f}
    </div>

    <div class="small-muted" style="margin-top:10px;">{explanation}</div>
  </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)



def render_tier_status_card(
    *,
    tier_name: str,
    tier_code: str,
    ci_conf: float,
    presence: float,
    stability: float,
    reliability: float,
    explanation: str = "",
) -> None:
    ICON = {
        "exploratory": "🔴",
        "conditional": "🟡",
        "admissible": "🟢",
    }.get(tier_code, "⚪")

    tier_color = TIER_COLORS.get(tier_code, TIER_COLORS["unknown"])

    st.markdown(
        f"""
<div class="tier-card">
  <div class="tier-strip" style="background:{tier_color['strip']};"></div>
  <div class="tier-body" style="background:{tier_color['bg']};">
    <div style="font-weight:700; font-size:1.05rem;">{ICON} {tier_name} (CI gating blocked)</div>
    <div class="small-muted" style="margin-top:4px;">
      CI Confidence {ci_conf:.3f} — presence {presence:.3f}, stability {stability:.3f}, reliability {reliability:.3f}
    </div>
     <div class="metric-grid">
    <div class="metric-tile"><div class="metric-label">CI Confidence</div><div class="metric-value">{ci_conf:.3f}</div></div>
    <div class="metric-tile"><div class="metric-label">Presence</div><div class="metric-value">{presence:.3f}</div></div>
    <div class="metric-tile"><div class="metric-label">Stability</div><div class="metric-value">{stability:.3f}</div></div>
    <div class="metric-tile"><div class="metric-label">Reliability</div><div class="metric-value">{reliability:.3f}</div></div>
        </div>
    <div class="small-muted" style="margin-top:8px;">{explanation}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )




    