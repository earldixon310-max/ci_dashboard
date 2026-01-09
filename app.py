# app.py — CI Report Viewer (story-first, no interactive map)
# - Satellite (GOES) + Radar (NWS mosaic) for visual context
# - NHC CurrentStorms for active tropical systems + advisory links
# - NWS Alerts near selected point (and optional state fallback)
# - CI summary + windows table driven by ci_report.json (global OR per-tile if spatial.tiles exists)

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import base64

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="CI Report Viewer", layout="wide")

st.markdown(
    """
<style>
:root { color-scheme: dark; }
.block-container { padding-top: 1.2rem; }

.small-muted { color: rgba(255,255,255,0.65); font-size: 0.92rem; }

.card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.tier-card {
  display:flex; gap:14px; align-items:stretch;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  overflow:hidden;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.tier-strip { width: 10px; }
.tier-body { padding: 14px 16px; width: 100%; }

.metric-grid {
  display:grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-top: 10px;
}
.metric-tile {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  padding: 12px 12px;
}
.metric-label { color: rgba(255,255,255,0.75); font-size: 0.9rem; }
.metric-value { font-size: 2.0rem; line-height: 1.1; margin-top: 6px; }

.pill {
  display:inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 0.5rem;
  font-size: 0.85rem;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(0,0,0,0.25);
}

hr.sep { border: none; height: 1px; background: rgba(255,255,255,0.08); margin: 18px 0; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Banner
# ----------------------------
from pathlib import Path
import streamlit as st

def hero_banner(
    title: str,
    *,
    image_path: str = "cyclone.png",
    height_px: int = 320,
    max_width_px: int = 1800,
) -> None:
    """
    Banner that works reliably even when Streamlit's working directory changes.
    Looks for the image next to this file first, then falls back to CWD.
    """
    here = Path(__file__).resolve().parent
    p1 = here / image_path
    p2 = Path(image_path)

    p = p1 if p1.exists() else p2
    if not p.exists():
        st.warning(f"Banner image not found: tried `{p1}` and `{p2}`")
        st.markdown(f"# {title}")
        # Gradient fallback
        st.markdown(
            f"""
            <div style="
                width: 100%;
                height: {height_px}px;
                border-radius: 22px;
                border: 1px solid rgba(255,255,255,0.08);
                background: radial-gradient(900px 300px at 30% 10%, rgba(70,120,170,0.35), rgba(0,0,0,0)),
                            radial-gradient(900px 300px at 70% 30%, rgba(160,80,150,0.26), rgba(0,0,0,0)),
                            rgba(255,255,255,0.02);
                box-shadow: 0 10px 40px rgba(0,0,0,0.35);
                position: relative;
                overflow: hidden;
                margin-bottom: 18px;
            ">
              <div style="
                position:absolute;
                left:22px; top:18px;
                color:white;
                font-size:40px;
                font-weight:800;
                text-shadow: 0 2px 16px rgba(0,0,0,0.55);
                letter-spacing: -0.02em;
              ">{title}</div>
              <div style="
                position:absolute;
                left:24px; top:72px;
                color: rgba(255,255,255,0.82);
                font-size:13px;
                font-weight:500;
                text-shadow: 0 2px 12px rgba(0,0,0,0.55);
              ">Confidence Index (CI) diagnostics + event context</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .ci-hero-wrap {{
            width: 100%;
            display: flex;
            justify-content: center;
        }}
        .ci-hero {{
            position: relative;
            width: 100%;
            max-width: {max_width_px}px;
            height: {height_px}px;
            border-radius: 14px;
            overflow: hidden;
            margin: 0.25rem 0 1rem 0;
            background-image: url("data:image/png;base64,{b64}");
            background-size: cover;
            background-position: center;
        }}
        .ci-hero::after {{
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(
                to bottom,
                rgba(0,0,0,0.55),
                rgba(0,0,0,0.20) 45%,
                rgba(0,0,0,0.65)
            );
        }}
        .ci-hero-title {{
            position: absolute;
            top: 16px;
            left: 18px;
            right: 18px;
            z-index: 2;
            color: white;
            font-size: 44px;
            font-weight: 800;
            letter-spacing: 0.5px;
            text-shadow: 0px 2px 18px rgba(0,0,0,0.65);
            line-height: 1.05;
        }}
        </style>

        <div class="ci-hero-wrap">
          <div class="ci-hero">
              <div class="ci-hero-title">{title}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# Helpers / constants
# ----------------------------
TIER_COLORS = {
    "exploratory": {"strip": "#b24a4a", "bg": "rgba(178,74,74,0.18)"},
    "conditional": {"strip": "#b89b2b", "bg": "rgba(184,155,43,0.18)"},
    "admissible": {"strip": "#2f9b5c", "bg": "rgba(47,155,92,0.18)"},
    "unknown": {"strip": "#666666", "bg": "rgba(255,255,255,0.06)"},
}
SEVERITY_ORDER = {"Extreme": 4, "Severe": 3, "Moderate": 2, "Minor": 1, "Unknown": 0, None: 0}


def clamp01(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:
            return None
        return max(0.0, min(1.0, v))
    except Exception:
        return None


def as_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:
            return None
        return v
    except Exception:
        return None


def normalize_lon(lon: float, west_positive: bool) -> float:
    if west_positive and lon > 0:
        return -abs(lon)
    return lon


def point_in_bbox(lat: float, lon: float, bbox: Dict[str, Any]) -> bool:
    return (
        bbox.get("lat_min") <= lat <= bbox.get("lat_max")
        and bbox.get("lon_min") <= lon <= bbox.get("lon_max")
    )


def select_ci_bundle(report: Dict[str, Any], lat: float, lon: float) -> Dict[str, Any]:
    global_bundle = {
        "source": "global",
        "tile_id": None,
        "metrics": report.get("metrics", {}) or {},
        "interpretation": report.get("interpretation", {}) or {},
        "windows": report.get("windows", []) or [],
        "counts": report.get("counts", {}) or {},
    }

    spatial = report.get("spatial") or {}
    tiles = spatial.get("tiles") or []
    if not tiles:
        return global_bundle

    selected = None
    for t in tiles:
        bbox = (t.get("bbox") or {})
        if bbox and point_in_bbox(lat, lon, bbox):
            selected = t
            break
    if not selected:
        return global_bundle

    t_metrics = (selected.get("metrics") or {})
    t_interp = (selected.get("interpretation") or {})
    t_windows = selected.get("windows")

    merged_metrics = dict(global_bundle["metrics"])
    merged_metrics.update(t_metrics)

    merged_interp = dict(global_bundle["interpretation"])
    merged_interp.update(t_interp)

    return {
        "source": "tile",
        "tile_id": selected.get("id"),
        "metrics": merged_metrics,
        "interpretation": merged_interp,
        "windows": t_windows if isinstance(t_windows, list) and len(t_windows) > 0 else global_bundle["windows"],
        "counts": global_bundle["counts"],
    }


def windows_to_df(windows: List[Dict[str, Any]], tile_id: Optional[str] = None) -> pd.DataFrame:
    rows = []
    for w in windows or []:
        rows.append(
            {
                "index": w.get("index"),
                "start": w.get("start"),
                "end": w.get("end"),
                "valid": bool(w.get("valid")),
                "tau_star": w.get("tau_star"),
                "corr": w.get("corr"),
                "delta": w.get("delta"),
                "validity_reason": w.get("validity_reason"),
                "tile_id": tile_id or w.get("tile_id"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        for c in ["corr", "delta"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_window_summary(dfw: pd.DataFrame) -> Dict[str, Any]:
    if dfw is None or dfw.empty:
        return {"N_windows": 0, "N_valid": 0, "invalid_start_end": 0, "missing_fields": "—"}
    N_windows = int(len(dfw))
    N_valid = int(dfw["valid"].sum()) if "valid" in dfw.columns else 0
    invalid_start_end = int(((dfw["start"].isna()) | (dfw["end"].isna())).sum()) if set(["start", "end"]).issubset(dfw.columns) else 0

    missing = []
    for f in ["tau_star", "corr", "delta"]:
        if f in dfw.columns:
            nmiss = int(dfw[f].isna().sum())
            if nmiss:
                missing.append(f"{f}={nmiss}")
    return {"N_windows": N_windows, "N_valid": N_valid, "invalid_start_end": invalid_start_end, "missing_fields": (", ".join(missing) if missing else "none")}


GOES_IMAGES = {
    "GOES-16 CONUS (Clean IR)": "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/13/1250x750.jpg",
    "GOES-16 CONUS (GeoColor)": "https://cdn.star.nesdis.noaa.gov/GOES16/ABI/CONUS/GEOCOLOR/1250x750.jpg",
    "GOES-18 CONUS (Clean IR)": "https://cdn.star.nesdis.noaa.gov/GOES18/ABI/CONUS/13/1250x750.jpg",
    "GOES-18 CONUS (GeoColor)": "https://cdn.star.nesdis.noaa.gov/GOES18/ABI/CONUS/GEOCOLOR/1250x750.jpg",
}
RADAR_IMAGES = {
    "NWS Radar CONUS (loop)": "https://radar.weather.gov/ridge/standard/CONUS_loop.gif",
    "NWS Radar Pacific (loop)": "https://radar.weather.gov/ridge/standard/PACN_loop.gif",
    "NWS Radar Southeast (loop)": "https://radar.weather.gov/ridge/standard/SOUTHEAST_loop.gif",
    "NWS Radar Northeast (loop)": "https://radar.weather.gov/ridge/standard/NORTHEAST_loop.gif",
}
REGION_PRESETS = {
    "West Coast (LA)": (34.0522, -118.2437, "CA"),
    "Pacific NW (Seattle)": (47.6062, -122.3321, "WA"),
    "Gulf Coast (New Orleans)": (29.9511, -90.0715, "LA"),
    "Florida (Miami)": (25.7617, -80.1918, "FL"),
    "Midwest (Chicago)": (41.8781, -87.6298, "IL"),
    "Northeast (NYC)": (40.7128, -74.0060, "NY"),
}


@st.cache_data(ttl=300)
def fetch_nhc_current_storms() -> Dict[str, Any]:
    url = "https://www.nhc.noaa.gov/CurrentStorms.json"
    headers = {"User-Agent": "CI-Report-Viewer (hsagconsortium.com)", "Accept": "application/json"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()


def nhc_storms_to_df(nhc_json: Dict[str, Any]) -> pd.DataFrame:
    storms = nhc_json.get("activeStorms", []) or []
    rows = []
    for s in storms:
        rows.append(
            {
                "id": s.get("id"),
                "name": s.get("name"),
                "classification": s.get("classification"),
                "intensity_kt": s.get("intensity"),
                "pressure_mb": s.get("pressure"),
                "lat": s.get("latitude_numeric"),
                "lon": s.get("longitude_numeric"),
                "lastUpdate": s.get("lastUpdate"),
                "public_adv": (s.get("publicAdvisory") or {}).get("url"),
                "forecast_adv": (s.get("forecastAdvisory") or {}).get("url"),
                "discussion": (s.get("forecastDiscussion") or {}).get("url"),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def fetch_url_text(url: str) -> str:
    """
    Fetch text from an advisory/discussion URL.
    Best-effort: strip basic HTML tags, keep it readable.
    """
    headers = {"User-Agent": "CI-Report-Viewer (hsagconsortium.com)"}
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()

    text = r.text

    # Strip scripts/styles (best effort)
    text = re.sub(r"<script.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Strip remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



@st.cache_data(ttl=300)
def fetch_nws_alerts_point(lat: float, lon: float) -> Dict[str, Any]:
    url = f"https://api.weather.gov/alerts/active?point={lat:.4f},{lon:.4f}"
    headers = {"User-Agent": "CI-Report-Viewer (hsagconsortium.com)", "Accept": "application/geo+json"}
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def fetch_nws_alerts_area(state_code: str) -> Dict[str, Any]:
    url = f"https://api.weather.gov/alerts/active?area={state_code}"
    headers = {"User-Agent": "CI-Report-Viewer (hsagconsortium.com)", "Accept": "application/geo+json"}
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()
    return r.json()


def nws_alerts_to_df(alerts_geojson: Dict[str, Any]) -> pd.DataFrame:
    feats = alerts_geojson.get("features", []) or []
    rows = []
    for f in feats:
        p = (f.get("properties") or {})
        rows.append(
            {
                "event": p.get("event"),
                "headline": p.get("headline"),
                "severity": p.get("severity"),
                "urgency": p.get("urgency"),
                "certainty": p.get("certainty"),
                "areaDesc": p.get("areaDesc"),
                "sent": p.get("sent"),
                "expires": p.get("expires"),
                "link": (p.get("@id") or p.get("id")),
            }
        )
    return pd.DataFrame(rows)


def summarize_alerts(df: pd.DataFrame) -> Tuple[int, str]:
    if df is None or df.empty:
        return 0, "None"
    max_sev, max_score = "Unknown", 0
    for s in df.get("severity", []):
        sc = SEVERITY_ORDER.get(s, 0)
        if sc > max_score:
            max_score, max_sev = sc, (s or "Unknown")
    return int(len(df)), str(max_sev)


# ----------------------------
# Load CI report
# ----------------------------
report_path = Path("ci_report.json")
if not report_path.exists():
    st.error("ci_report.json not found in this folder.")
    st.stop()
with report_path.open("r", encoding="utf-8") as f:
    report = json.load(f)


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Story controls")
st.sidebar.caption("Pick a weather story first. CI metrics provide the admissibility signal for that story.")

goes_choice = st.sidebar.selectbox("Satellite imagery (GOES)", list(GOES_IMAGES.keys()), index=1)
show_goes = st.sidebar.checkbox("Show satellite image", value=True)
show_radar = st.sidebar.checkbox("Show radar mosaic (NWS)", value=True)
radar_choice = st.sidebar.selectbox("Radar mosaic", list(RADAR_IMAGES.keys()), index=0, disabled=not show_radar)

st.sidebar.markdown("---")
context_mode = st.sidebar.radio("Context mode", ["Global CI report", "Event-driven (NHC/NWS)"], index=1)

preset_name = st.sidebar.selectbox("Region preset (for NWS alerts)", list(REGION_PRESETS.keys()), index=0)
preset_lat, preset_lon, preset_state = REGION_PRESETS[preset_name]

west_positive = st.sidebar.checkbox("I entered West longitude as positive", value=False)

manual_override = st.sidebar.checkbox("Manual lat/lon override", value=False)
if manual_override:
    lat_in = st.sidebar.number_input("Latitude", value=float(preset_lat), min_value=-90.0, max_value=90.0, step=0.01, format="%.4f")
    lon_in = st.sidebar.number_input("Longitude", value=float(preset_lon), min_value=-180.0, max_value=180.0, step=0.01, format="%.4f")
else:
    lat_in, lon_in = float(preset_lat), float(preset_lon)

lat = float(lat_in)
lon = float(normalize_lon(float(lon_in), west_positive))

st.sidebar.caption(f"Using alerts point: {lat:.4f}, {lon:.4f}  (preset: {preset_name})")

st.sidebar.markdown("---")
show_nhc = st.sidebar.checkbox("Show NHC active storms", value=True)
show_nws = st.sidebar.checkbox("Show NWS alerts near point", value=True)
if st.sidebar.button("Refresh external feeds"):
    st.cache_data.clear()
    st.rerun()


# ----------------------------
# Main
# ----------------------------
hero_banner("CI Report Viewer", image_path="cyclone.png", height_px=210, max_width_px=1800)

left, right = st.columns([1.15, 0.85], gap="large")

with left:
 #   st.markdown("## Event context")

 #   if show_goes:
 #       st.image(GOES_IMAGES[goes_choice], use_container_width=True)
  #      st.caption(f"{goes_choice} (latest frame).")

 #   if show_radar:
 #       st.image(RADAR_IMAGES[radar_choice], use_container_width=True)
  #      st.caption(f"{radar_choice} (animated loop).")
#with st.container(border=True):
    st.markdown("## Coherence Index (CI): What You’re Seeing")
    st.markdown(
        """
**What CI is measuring**  
CI measures whether large-scale environmental signals are structured, repeatable, and statistically non-random across time.

**Operational use**  
CI is a gating layer: it indicates when downstream interpretation is trustworthy.

**BLOCKED vs ENABLED**  
- **BLOCKED**: coherence is insufficient for operational confidence  
- **ENABLED**: coherence supports decisive interpretation

**Tiers**  
- **Exploratory**: situational awareness  
- **Conditional**: limited but usable support  
- **Admissible**: operational-grade confidence

**What to look for**  
Use CI alongside alerts + NHC + imagery. When they align, confidence is highest.
        """
    )


    st.markdown('<hr class="sep"/>', unsafe_allow_html=True)

    if show_nhc:
        st.markdown("### Active tropical systems (NHC)")
        try:
            nhc = fetch_nhc_current_storms()
            df_st = nhc_storms_to_df(nhc)
            if df_st.empty:
                st.info("No active tropical systems reported by NHC right now.")
            else:
                st.dataframe(df_st[["name", "classification", "intensity_kt", "pressure_mb", "lat", "lon", "lastUpdate"]], use_container_width=True, hide_index=True)

                storm_names = (df_st["name"].fillna("Unnamed") + " — " + df_st["classification"].fillna("")).tolist()
                pick = st.selectbox("Select storm for advisory links", storm_names)
                i = storm_names.index(pick)
                row = df_st.iloc[i].to_dict()

                cols = st.columns(3)
                if row.get("public_adv"):
                    cols[0].markdown(f"[Public advisory]({row['public_adv']})")
                if row.get("forecast_adv"):
                    cols[1].markdown(f"[Forecast advisory]({row['forecast_adv']})")
                if row.get("discussion"):
                    cols[2].markdown(f"[Discussion]({row['discussion']})")

                with st.expander("Show discussion text (best-effort parse)", expanded=False):
                    disc_url = row.get("discussion")
                    if disc_url:
                        try:
                            txt = fetch_url_text(disc_url)
                            st.text(txt[:6000] + ("\n\n…(truncated)" if len(txt) > 6000 else ""))
                        except Exception as e:
                            st.warning(f"Could not fetch/parse discussion text: {e}")
                    else:
                        st.info("No discussion URL provided for this system.")
        except Exception as e:
            st.error(f"NHC feed failed: {e}")

    st.markdown('<hr class="sep"/>', unsafe_allow_html=True)

    if show_nws:
        st.markdown("### Watches / Warnings / Advisories (NWS)")
        st.caption("Radar/precipitation does *not* guarantee an official alert. Alerts appear only when NWS issues them for your selected point/area.")

        try:
            alerts_geo = fetch_nws_alerts_point(lat, lon)
            df_a = nws_alerts_to_df(alerts_geo)
            count, max_sev = summarize_alerts(df_a)

            if count == 0:
                st.info("No active alerts at this exact point. Showing state-wide alerts as fallback.")
                alerts_geo2 = fetch_nws_alerts_area(preset_state)
                df_a2 = nws_alerts_to_df(alerts_geo2)
                count2, max_sev2 = summarize_alerts(df_a2)
                st.write(f"State: **{preset_state}** — Alerts: **{count2}**, Max severity: **{max_sev2}**")
                if df_a2.empty:
                    st.info("No active NWS alerts returned for this state.")
                else:
                    show_cols = ["event", "severity", "urgency", "headline", "areaDesc", "sent", "expires", "link"]
                    st.dataframe(df_a2[show_cols], use_container_width=True, hide_index=True)
            else:
                st.write(f"Point alerts — Alerts: **{count}**, Max severity: **{max_sev}**")
                show_cols = ["event", "severity", "urgency", "headline", "areaDesc", "sent", "expires", "link"]
                st.dataframe(df_a[show_cols], use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"NWS alerts fetch failed: {e}")


with right:
    st.subheader("CI Summary")

    bundle = select_ci_bundle(report, lat, lon)
    metrics = bundle.get("metrics", {}) or {}
    interp = bundle.get("interpretation", {}) or {}

    tier = (interp.get("ci_tier") or {}) if isinstance(interp.get("ci_tier"), dict) else {}
    tier_name = tier.get("name") or interp.get("tier_name") or "Exploratory"
    tier_code = (tier.get("code") or interp.get("tier_code") or "exploratory").lower()
    tier_color = TIER_COLORS.get(tier_code, TIER_COLORS["unknown"])

    tier_desc = {
        "exploratory": "Early / weak structure. Use for research, not gating downstream forecasts.",
        "conditional": "Moderate structure. Use with caution; combine with other signals.",
        "admissible": "Strong structure. CI may be used to gate/inform downstream forecast logic.",
        "unknown": "Tier unknown.",
    }.get(tier_code, "Tier unknown.")

    st.markdown(
        f"""
<div class="tier-card">
  <div class="tier-strip" style="background:{tier_color['strip']};"></div>
  <div class="tier-body" style="background:{tier_color['bg']};">
    <div style="font-weight:700; font-size:1.05rem;">{tier.get('icon','●')} {tier_name} (CI tier)</div>
    <div class="small-muted" style="margin-top:4px;">{tier_desc}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    global_m = report.get("metrics", {}) or {}
    ci_conf = clamp01(metrics.get("ci_confidence"))
    presence = clamp01(metrics.get("presence"))
    stability = clamp01(metrics.get("stability"))
    reliability = clamp01(metrics.get("reliability"))

    def _fallback(v, key):
        return v if v is not None else clamp01(global_m.get(key))

    ci_conf = _fallback(ci_conf, "ci_confidence")
    presence = _fallback(presence, "presence")
    stability = _fallback(stability, "stability")
    reliability = _fallback(reliability, "reliability")

    st.markdown(
        f"""
<div class="metric-grid">
  <div class="metric-tile"><div class="metric-label">CI Confidence</div><div class="metric-value">{(ci_conf or 0.0):.3f}</div></div>
  <div class="metric-tile"><div class="metric-label">Presence</div><div class="metric-value">{(presence or 0.0):.3f}</div></div>
  <div class="metric-tile"><div class="metric-label">Stability</div><div class="metric-value">{(stability or 0.0):.3f}</div></div>
  <div class="metric-tile"><div class="metric-label">Reliability</div><div class="metric-value">{(reliability or 0.0):.3f}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    gating_threshold = as_float_or_none(interp.get("gating_threshold"))
    gating_decision = (interp.get("gating_decision") or interp.get("decision") or "—")
    explanation = interp.get("explanation") or ""

    st.write("")
    st.write(f"**Decision:** <span class='pill'>{gating_decision}</span>", unsafe_allow_html=True)
    st.write(f"**Threshold:** <span class='pill'>{gating_threshold if gating_threshold is not None else '—'}</span>", unsafe_allow_html=True)

    if bundle.get("source") == "tile":
        st.caption(f"CI shown is **tile-level** for this point (tile: `{bundle.get('tile_id')}`).")
    else:
        st.caption("CI shown is the **global (report-level)** summary. (Add `spatial.tiles` for per-location CI.)")

    if explanation:
        st.caption(explanation)

    with st.expander("What do these mean? (viewer-facing definitions)", expanded=False):
        st.markdown(
            """
- **CI Confidence**: a *pre-forecast admissibility score* (0–1) summarizing whether CI-derived structure is trustworthy enough to influence downstream forecasting.
- **Presence**: fraction of windows that pass CI validity checks.
- **Stability**: consistency of inferred lag **τ\*** across valid windows (stable τ\* → closer to 1).
- **Reliability**: median Δ vs null expectation (clipped to 0–1).
- **Decision (BLOCKED/ENABLED)**: whether CI-derived constraints are allowed into downstream forecast logic (gating).
- **Threshold**: the CI Confidence cutoff used for the gate decision.
- **CI tier**: a human-readable band summarizing admissibility posture (exploratory → conditional → admissible).
            """
        )

    st.markdown('<hr class="sep"/>', unsafe_allow_html=True)

    st.subheader("Window summary")
    dfw = windows_to_df(bundle.get("windows", []) or [], tile_id=bundle.get("tile_id"))
    summ = compute_window_summary(dfw)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='card'><div class='metric-label'>N_windows</div><div class='metric-value'>{summ['N_windows']}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><div class='metric-label'>N_valid</div><div class='metric-value'>{summ['N_valid']}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'><div class='metric-label'>Invalid start/end</div><div class='metric-value'>{summ['invalid_start_end']}</div></div>", unsafe_allow_html=True)

    st.caption(f"Missing fields in current selection: {summ['missing_fields']}")

    if dfw.empty:
        st.info("No CI windows available for this selection.")
    else:
        st.dataframe(dfw, use_container_width=True, hide_index=True)