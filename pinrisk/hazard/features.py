"""Feature engineering: raw data layers -> one tidy table, one row per cell.

This produces data/processed/features.csv — the "grid contract" every later
module reads. Columns:

  identity   : cell_id, i, j, lon, lat, pincode, pincode_name
  terrain    : elevation_m, slope_deg, hand_m, dist_drainage_m, dist_coast_m
  surface    : imperviousness, landuse, built_area_m2
  forcing    : rainfall_2015_mm
  label      : flooded_2015 (1 = inside the observed 2015 flood extent)

Feature glossary (the WHY of each predictor):
  elevation_m     low ground collects water.
  slope_deg       flat cells drain slowly; steep cells shed water.
  hand_m          Height Above Nearest Drainage: metres a cell sits above the
                  nearest river/canal. THE classic flood predictor — a cell
                  1 m above the Adyar floods long before one 20 m above it,
                  even at identical absolute elevation.
                  MVP simplification: our HAND uses straight-line nearest
                  drainage, not the flow-path-based HAND (pysheds/WhiteboxTools
                  compute the real thing from flow direction grids).
  dist_drainage_m fluvial risk decays away from channels.
  dist_coast_m    coastal/surge influence proxy (secondary for this peril).
  imperviousness  paved ground turns rain into instant runoff (pluvial driver).
  rainfall_2015_mm event rainfall — near-uniform at city scale (25 km native
                  resolution!); terrain provides the local variation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ensure_dirs
from ..grid import base_table, cell_area_m2, cell_size_m, raster_to_col
from ..datasources import real


def _load_raster(raw: Path, name: str) -> np.ndarray:
    return np.load(raw / f"{name}.npy")


def build_features(cfg: dict) -> pd.DataFrame:
    """Assemble the per-cell feature table and write features.csv."""
    ensure_dirs(cfg)
    raw = Path(cfg["paths"]["raw"])
    processed = Path(cfg["paths"]["processed"])

    meta = json.loads((raw / "meta.json").read_text())
    pincode_list = meta["pincodes"]

    # ---- load layers, preferring real files when the user has added them ----
    def layer(name: str, synthetic_name: str | None = None) -> np.ndarray:
        real_arr = real.load_layer(cfg, name)
        if real_arr is not None:
            print(f"  [features] using REAL data for '{name}'")
            return real_arr
        return _load_raster(raw, synthetic_name or name)

    dem = layer("dem")
    rainfall = layer("rainfall_event")
    flood = layer("flood_extent_2015")
    imperv = layer("imperviousness")
    is_sea = _load_raster(raw, "is_sea").astype(bool)
    dist_drainage = _load_raster(raw, "dist_drainage")
    dist_coast = _load_raster(raw, "dist_coast")
    drain_elev = _load_raster(raw, "drain_elev")
    landuse_code = _load_raster(raw, "landuse_code").astype(int)
    pincode_idx = _load_raster(raw, "pincode_idx").astype(int)

    # Built area: real layer is m2/cell directly; synthetic stores a density.
    real_built = real.load_layer(cfg, "building_area")
    if real_built is not None:
        built_area = real_built
        print("  [features] using REAL data for 'building_area'")
    else:
        built_area = _load_raster(raw, "building_density") * cell_area_m2(cfg)

    # ---- derived terrain features ----------------------------------------
    # Slope from the DEM gradient. Sea filled with coastal elevation so the
    # coastline doesn't create a fake cliff of steep slopes.
    cell_h, cell_w = cell_size_m(cfg)
    dem_filled = np.where(is_sea, 0.3, dem)
    gy, gx = np.gradient(dem_filled, cell_h, cell_w)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

    # HAND-lite: cell elevation minus elevation of its nearest drainage point.
    hand = np.clip(dem - drain_elev, 0.0, None)

    # ---- assemble the table (LAND CELLS ONLY — the sea is not insurable) --
    df = base_table(cfg)
    keep = ~raster_to_col(df, is_sea.astype(float)).astype(bool)
    df = df[keep].reset_index(drop=True)

    landuse_names = np.array(["other", "urban_sparse", "urban_dense"])
    pin_codes = np.array([p["pincode"] for p in pincode_list])
    pin_names = np.array([p["name"] for p in pincode_list])

    for col, arr in [
        ("elevation_m", dem),
        ("slope_deg", slope_deg),
        ("hand_m", hand),
        ("dist_drainage_m", dist_drainage),
        ("dist_coast_m", dist_coast),
        ("imperviousness", imperv),
        ("built_area_m2", built_area),
        ("rainfall_2015_mm", rainfall),
        ("flooded_2015", flood),
    ]:
        df[col] = raster_to_col(df, arr)
    df["flooded_2015"] = df["flooded_2015"].astype(int)
    lu_idx = raster_to_col(df, landuse_code.astype(float)).astype(int)
    df["landuse"] = landuse_names[lu_idx]
    pi = raster_to_col(df, pincode_idx.astype(float)).astype(int)
    df["pincode"] = pin_codes[pi]
    df["pincode_name"] = pin_names[pi]

    out = processed / "features.csv"
    df.to_csv(out, index=False)
    n_flooded = int(df["flooded_2015"].sum())
    print(
        f"  [features] {len(df):,} land cells | {n_flooded:,} flooded in 2015 "
        f"({100 * n_flooded / len(df):.1f}% — this class imbalance is why we "
        f"use class weights + PR-AUC, never plain accuracy)"
    )
    print(f"  [features] wrote {out}")
    return df
