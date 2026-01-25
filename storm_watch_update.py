# storm_watch_update.py
from __future__ import annotations

import argparse
from pathlib import Path

from storm_watch_core import (
    ensure_sw_shape,
    load_json,
    save_json,
    append_update,
    simulate_next_update,
    ci_report_to_storm_update,
)

def main() -> int:
    p = argparse.ArgumentParser(description="Append updates to storm_watch.json (CI-derived or simulated).")
    p.add_argument("--storm-watch", type=str, default="storm_watch.json", help="Path to storm_watch.json")
    p.add_argument("--ci-report", type=str, default="ci_report.json", help="Path to ci_report.json")
    p.add_argument("--tile", type=str, default=None, help="Tile id (default GLOBAL)")
    p.add_argument("--tminus", type=float, default=None, help="t_minus_h for the update")
    p.add_argument("--mode", type=str, choices=["ci", "simulate"], default="ci", help="Update mode")
    args = p.parse_args()

    sw_path = Path(args.storm_watch).resolve()
    ci_path = Path(args.ci_report).resolve()

    if sw_path.exists():
        sw = ensure_sw_shape(load_json(sw_path))
    else:
        # Create a minimal storm watch file if missing
        sw = ensure_sw_shape({
            "target": {"name": "Unnamed", "zone_id": "—"},
            "event": {},
            "updates": [],
        })

    if args.mode == "simulate":
        u = simulate_next_update(sw)
        if args.tminus is not None:
            u["t_minus_h"] = args.tminus
        sw = append_update(sw, u)
    else:
        # CI-derived update
        tile_id = None if (args.tile in (None, "", "GLOBAL")) else args.tile
        u = ci_report_to_storm_update(
            ci_report_path=ci_path,
            tile_id=tile_id,
            t_minus_h=args.tminus,
        )
        sw = append_update(sw, u)

    save_json(sw_path, sw)
    print(f"OK: appended update -> {sw_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
