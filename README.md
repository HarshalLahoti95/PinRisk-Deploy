# PinRisk MVP — Pincode-level Urban Flood Risk (Chennai)

A thin, end-to-end, **runnable** catastrophe-model pipeline: for every Chennai
PIN code it produces a flood **hazard score (0–100)**, an **expected annual
loss (AAL, ₹)**, a **confidence label**, and the **data sources** behind each
number — backtested against the December 2015 Chennai flood and shown on an
interactive map dashboard.

> **Honesty first:** out of the box this runs on **clearly-labelled synthetic
> sample data** (realistic in shape, fake in value). Every output — console,
> metrics file, dashboard, validation figure — carries a synthetic-data
> disclaimer until you swap in real datasets (instructions below). Validation
> numbers on synthetic data prove the *plumbing works*, *not* real-world
> accuracy: the fake flood extent is generated from (mostly) the same terrain
> layers the model trains on, so metrics are optimistic by construction.

---

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python run_pipeline.py all      # ~1 minute: data → model → ₹ → backtest
.venv/bin/streamlit run dashboard.py      # opens the dashboard in your browser
```

Outputs land in:

| File | What it is |
|---|---|
| `outputs/pincode_risk.geojson` / `.csv` | per-pincode scores, AAL, confidence, drivers, sources |
| `outputs/validation/metrics.json` | backtest metrics (incl. the no-ML baseline) |
| `outputs/validation/predicted_vs_actual.png` | the 3-panel proof figure |
| `outputs/validation/pincode_validation.csv` | actual vs predicted flooded fraction per pincode |
| `data/processed/features.csv` | the per-cell feature table (the "grid contract") |
| `data/raw/provenance.json` | where every layer came from, real vs synthetic |

---

## How it works — the 4-module cat model

Insurance catastrophe models all share one structure; each module is a
separate file and they communicate **only** through the per-cell table
(`data/processed/features.csv`, one row per ~110 m grid cell):

```
hazard        exposure       vulnerability     financial
"P(flood)?"   "₹ at risk?"   "depth→%damage"   "multiply, integrate, aggregate"
pinrisk/      pinrisk/       pinrisk/          pinrisk/
hazard/*.py   exposure.py    vulnerability.py  financial.py
     │             │               │                │
     └─────────────┴───────────────┴────────────────┘
              every module reads/writes the same grid table
              → swap a module (new peril, new city) without touching the rest
```

Supporting cast: `grid.py` (the grid contract + spatial-block CV),
`provenance.py` (source tracking), `uncertainty.py` (confidence scoring),
`validation.py` (the permanent backtest harness),
`datasources/real.py` + `datasources/synthetic.py` (data acquisition).

### Hazard: the susceptibility model
Two classifiers are trained on the 2015 flood extent — penalised **logistic
regression** (every learned weight explainable) and **gradient-boosted trees**
(accuracy ceiling; physically constrained so risk can never *rise* with
elevation or *fall* with rainfall) — and averaged. Chosen over deep learning
deliberately: ~15 % positive labels from a single event would overfit a CNN,
and the team maintaining this is not an ML-research team.

**"What weights do elevation vs rainfall get?"** — learned from data, never
hand-assigned. See `learned_weights_logistic` in
`data/processed/hazard_metrics.json` after a run.

### Validation: spatially-blocked, always vs a trivial baseline
Random train/test splits cheat on spatial data (neighbouring cells are
near-copies → "spatial leakage"). We hold out whole ~2 km blocks instead, and
every reported metric uses **out-of-fold** predictions. A **no-ML baseline**
(rank cells by height-above-drainage) is always reported alongside; if the ML
doesn't beat it, the pipeline says so loudly. Plain accuracy is never shown
(15 % prevalence makes it meaningless) — use ROC-AUC, PR-AUC, Brier, CSI.

### Financial: from probability to ₹
`E[loss | RP event] = P(flood) × exposure ₹ × damage_ratio(depth)`, then AAL
= trapezoid integral of losses over annual exceedance probability (1/RP) for
RP ∈ {10, 50, 100}. Hazard score (0–100) = 100 × exposure-weighted mean
P(flood in the 100-year event) per pincode.

### Climate scenarios (illustrative)
The dashboard's 2050 toggle uses the **delta-change method**: multiply the
rainfall field (×1.10 / ×1.22), push it through the *same* trained model.
The method is standard; the multipliers are **placeholders** — production
values must be derived from NEX-GDDP-CMIP6 / CORDEX South Asia and shown as
ranges. Everything downstream of the rainfall shift is already wired.

---

## Swapping in real data

Drop files with these exact names into `data/raw/real/`, then re-run
`python run_pipeline.py all`. Anything you don't provide stays synthetic
(and stays labelled). Full instructions incl. URLs and Earth Engine snippets:
`pinrisk/datasources/real.py` (`REAL_SOURCES`).

| Layer | File | Real source |
|---|---|---|
| Elevation | `dem.tif` | FABDEM v1.2 (~30 m) |
| Event rainfall | `rainfall_event.tif` | IMD gridded 0.25° (Dec 2015 total) |
| 2015 flood extent | `flood_extent_2015.tif` | Sentinel-1 SAR / Global Flood Database |
| Imperviousness | `imperviousness.tif` | ESA WorldCover 10 m |
| Building area | `building_area.tif` | Microsoft + Google footprints, rasterised |
| Drainage lines | `drainage.geojson` | HydroRIVERS + OSM waterways |
| Pincode polygons | `pincode_boundaries.geojson` | datameet / data.gov.in derived |

GeoTIFF loading needs `pip install rasterio` (any CRS/resolution; it is
reprojected and resampled onto the analysis grid automatically).

**The pincode-boundary gap is real:** India Post publishes no official
pincode polygons. Until you add community-derived ones, boundaries are
Voronoi approximations around locality centroids and are labelled as such.

---

## Every assumption in one place

All tunable numbers live in `config.yaml` with comments. The big ones:

1. **Asset values** (₹/m² by land-use class + contents factor) — invented,
   configurable; an insurer's portfolio replaces them in production.
2. **Depth-damage curve** — JRC-style Asia residential shape (Huizinga et al.
   2017), approximately digitised. Verify before external use.
3. **Depth-if-flooded** — `ref_depth(RP) · exp(−HAND/2 m)` heuristic, NOT a
   hydraulic simulation.
4. **Rainfall frequency** — the 2015 event ≈ 1-in-100-year rainfall, with
   fixed scaling to RP50/RP10. Replace with IMD frequency analysis.
5. **HAND-lite** — straight-line distance to the nearest channel, not
   flow-path HAND (use `pysheds` for the real thing).
6. **Rainfall resolution** — native ~25 km: rainfall is near-uniform at city
   scale and terrain provides all local variation. A physical limit worth
   stating in any pitch, not a bug.
7. **AAL truncation** — zero loss below RP10, flat tail beyond RP100, 3-point
   integration (production uses 10–50 RPs).
8. **Confidence** — model-disagreement heuristic capped at 0.5 on synthetic
   data; not a calibrated interval (see next steps).

---

## Interpreting the backtest

`outputs/validation/metrics.json` compares four rankings against the 2015
extent (out-of-fold): `trivial_hand_baseline`, `logistic`, `gbm`, `ensemble`.
The claim worth making is *relative*: **the model beats physics-only ranking
on held-out areas**. On synthetic data even that claim is about plumbing —
with real data it becomes your core demo artifact
(`predicted_vs_actual.png`: predicted probability | observed extent |
hit/false-alarm/miss agreement map).

If the baseline is weak (it may be — the MVP favours honesty over tuning):
say so, and improve via the next-steps list rather than threshold-shopping.

---

## Project layout

```
config.yaml            every assumption, commented
run_pipeline.py        CLI: data | features | hazard | financial | validate | all
dashboard.py           Streamlit app (map, drill-down, backtest, provenance)
pinrisk/
  grid.py              the analysis-grid contract + spatial blocking
  provenance.py        per-layer source registry (real vs synthetic)
  datasources/real.py  real-dataset catalogue + GeoTIFF loader
  datasources/synthetic.py  labelled synthetic stand-ins (this file = the fake city)
  hazard/features.py   rasters → per-cell feature table
  hazard/model.py      susceptibility models + blocked CV (the ML core)
  exposure.py          built area × ₹/m² assumptions
  vulnerability.py     depth-damage curve + depth heuristic
  financial.py         expected loss, AAL, pincode aggregation, drivers
  uncertainty.py       confidence scoring
  validation.py        the permanent backtest harness
data/raw/              inputs (synthetic .npy + real/ drop-zone)
data/processed/        features.csv, models, OOF predictions, metrics
outputs/               pincode_risk.geojson/.csv, validation artifacts
```

---

## Top 5 next steps to production credibility

1. **Real data end-to-end, validated for real.** Sentinel-1-derived 2015
   extent as labels, FABDEM terrain, IMD rainfall — then re-run this same
   harness and publish the honest numbers. Everything else is second to this.
2. **A real depth model.** Replace the depth-if-flooded heuristic with at
   least a bathtub/HAND-fill model, ideally LISFLOOD-FP on the 2015 event —
   depth drives damage, damage drives ₹.
3. **India-calibrated damage curves + real exposure.** Partner-insurer claims
   from 2015 Chennai to fit local depth-damage curves; their portfolio as
   exposure. This is the moat incumbents can't copy quickly.
4. **Real pincode boundaries + more events.** Adopt/refine community pincode
   polygons; add Mumbai 2005 (reconstructed extent) and Kerala 2018 as
   further backtests so validation stops being single-event.
5. **Calibrated uncertainty + climate deltas.** Conformal prediction or
   quantile forests for honest per-pincode intervals; NEX-GDDP-CMIP6-derived
   rainfall deltas (as ranges) to make the 2050 toggle defensible.
```
