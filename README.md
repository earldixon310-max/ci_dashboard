[README.md](https://github.com/user-attachments/files/24511834/README.md)
# Constraint Intelligence (CI) — Confidence Reporting & Forecast Gating

CI is a lightweight, pre-forecast admissibility layer that converts windowed detection diagnostics into a
single confidence score and a gating decision (ENABLED/BLOCKED). It is designed to be:
- deterministic (same inputs → same report),
- auditable (per-window records preserved),
- simple (small number of metrics with transparent definitions).

## What CI produces
CI writes a JSON artifact (default: `ci_report.json`) containing:
- counts (windows, valid windows),
- metrics (presence, stability, reliability, CI confidence),
- interpretation (gating threshold + decision),
- windows (primary channel),
- optionally `channels` for multi-channel audit (raw/hp).

A Streamlit viewer (`app.py`) renders `ci_report.json`.

---

## Quick start

### 1) Generate a report
```powershell
python generate_ci_report.py
