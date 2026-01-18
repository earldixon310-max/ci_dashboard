# lib/ci_io.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

from pathlib import Path
import json

def load_report(primary="ci_report.json", fallback="ci_report.example.json"):
    p = Path(primary)
    if not p.exists():
        p = Path(fallback)

    if not p.exists():
        raise FileNotFoundError(
            f"CI report not found. Expected '{primary}' (local) or '{fallback}' (repo)."
        )

    with p.open("r", encoding="utf-8") as f:
        return json.load(f), str(p)


def available_tiles(report):
    """
    Return list of tile dicts if spatial tiles exist, else [].
    Supports:
      report["spatial"]["tiles"] = {tile_id: {...}, ...}
      report["spatial"]["tiles"] = [{ "id": "...", ...}, ...]
    """
    spatial = report.get("spatial") or {}
    tiles = spatial.get("tiles")
    if not tiles:
        return []

    if isinstance(tiles, dict):
        out = []
        for tid, bundle in tiles.items():
            if isinstance(bundle, dict):
                b = dict(bundle)
                b.setdefault("id", tid)
                out.append(b)
        return out

    if isinstance(tiles, list):
        return [t for t in tiles if isinstance(t, dict) and "id" in t]

    return []


def select_bundle(report: Dict[str, Any], tile_id: str | None = None) -> Dict[str, Any]:
    """
    Returns either:
      - a global bundle (report-level) if tile_id is None or not found
      - a tile bundle if tile_id matches a tile
    """
    global_bundle = {
        "source": "global",
        "tile_id": None,
        "counts": report.get("counts", {}) or {},
        "metrics": report.get("metrics", {}) or {},
        "interpretation": report.get("interpretation", {}) or {},
        "windows": report.get("windows", []) or [],
        "timestamp_utc": report.get("timestamp_utc"),
        "ci_version": report.get("ci_version"),
    }

    if not tile_id:
        return global_bundle

    for t in available_tiles(report):
        if t.get("id") == tile_id:
            return {
                "source": "tile",
                "tile_id": t.get("id"),
                "counts": t.get("counts", {}) or {},
                "metrics": t.get("metrics", {}) or {},
                "interpretation": t.get("interpretation", {}) or {},
                # Prefer tile windows if present; else fall back to global
                "windows": (t.get("windows") or []) or global_bundle["windows"],
                "timestamp_utc": report.get("timestamp_utc"),
                "ci_version": report.get("ci_version"),
            }

    return global_bundle


