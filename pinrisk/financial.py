"""FINANCIAL module — hazard x exposure x vulnerability -> expected loss.

THE CORE EQUATION, per cell, per return-period (RP) scenario:

    E[loss | RP event] = P(cell floods | RP rainfall)          <- hazard model
                       x exposure_value_inr                    <- exposure
                       x damage_ratio(depth_if_flooded(RP))    <- vulnerability

No double counting: the hazard model's probability is SPATIAL ("given this
rainfall happens, does THIS cell flood?"); the return period supplies the
TEMPORAL frequency ("how often does this rainfall happen?").

AAL (Average Annual Loss) — the number insurers price from — integrates
scenario losses over annual exceedance probability p = 1/RP (trapezoid rule):

    AAL ~= sum over adjacent RPs of (p_i - p_{i+1}) * (L_i + L_{i+1}) / 2
           + p_last * L_last

Truncation assumptions (both stated, both config-visible):
  * events milder than the smallest RP (10y) contribute zero loss;
  * events beyond the largest RP (100y) contribute at the 100y level
    (a conservative floor — real tails are fatter).
With 3 RPs the integral is coarse; production models use 10-50 RPs.

Climate scenarios reuse the same machinery: multiply the rainfall field by
the (ILLUSTRATIVE) delta-change factor from config.yaml, re-predict, re-integrate.

The module also aggregates cells -> pincodes and derives the 0-100 hazard
score: 100 x exposure-weighted mean P(flood) in the 100-year baseline
scenario — i.e. "what share of this pincode's insured value is expected to
be inside the flood, in the big event?"
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .hazard.model import FEATURES, predict
from .vulnerability import damage_ratio, depth_if_flooded

# Human-readable phrases for the "top risk drivers" explanation.
# Maps (feature, direction that raises risk) -> phrase.
DRIVER_PHRASES = {
    ("elevation_m", -1): "low-lying terrain",
    ("slope_deg", -1): "flat ground (slow drainage)",
    ("hand_m", -1): "sits barely above nearest drainage (low HAND)",
    ("dist_drainage_m", -1): "close to a river/canal",
    ("dist_coast_m", -1): "close to the coast",
    ("imperviousness", 1): "heavily paved/built-up (fast runoff)",
    ("rainfall_mm", 1): "high event rainfall",
}


def _scenario_losses(cfg: dict, df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Per-cell flood probability & expected loss for every (climate, RP)."""
    rps = cfg["scenarios"]["return_periods"]
    rain_scale = cfg["scenarios"]["rainfall_scale"]
    for scen_key, scen in cfg["climate"]["scenarios"].items():
        mult = float(scen["rain_multiplier"])
        for rp in rps:
            rain = df["rainfall_2015_mm"].to_numpy() * rain_scale[rp] * mult
            p = predict(models, df, rain)
            depth = depth_if_flooded(df["hand_m"].to_numpy(), rp, cfg)
            dmg = damage_ratio(depth, cfg)
            df[f"p_flood_rp{rp}_{scen_key}"] = p["ensemble"]
            df[f"loss_rp{rp}_{scen_key}"] = (
                p["ensemble"] * df["exposure_value_inr"].to_numpy() * dmg
            )
            if scen_key == "baseline" and rp == max(rps):
                # Kept for uncertainty.py: model disagreement = cheap UQ.
                df["p_rp100_logistic"] = p["logistic"]
                df["p_rp100_gbm"] = p["gbm"]
    return df


def _aal(cfg: dict, df: pd.DataFrame, scen_key: str) -> np.ndarray:
    """Trapezoid integration of the loss-exceedance curve (see module doc)."""
    rps = sorted(cfg["scenarios"]["return_periods"])           # e.g. [10, 50, 100]
    probs = [1.0 / rp for rp in rps]                            # [0.1, 0.02, 0.01]
    losses = [df[f"loss_rp{rp}_{scen_key}"].to_numpy() for rp in rps]
    aal = np.zeros(len(df))
    for k in range(len(rps) - 1):
        aal += (probs[k] - probs[k + 1]) * 0.5 * (losses[k] + losses[k + 1])
    aal += probs[-1] * losses[-1]                               # tail floor
    return aal


def _top_drivers(cfg: dict, df: pd.DataFrame, models: dict, pin_df: pd.DataFrame) -> pd.Series:
    """Explain each pincode's risk via the logistic model's contributions.

    contribution(feature) = learned std. coefficient x pincode's z-scored mean.
    Positive contributions push risk UP; we phrase the top three. This is a
    linear-model attribution (fast, exact for the logistic member); SHAP on
    the GBM is the post-MVP upgrade.
    """
    pipe = models["logistic"]
    scaler, clf = pipe.named_steps["scale"], pipe.named_steps["clf"]
    X_mean = df.groupby("pincode")[
        [f for f in FEATURES if f != "rainfall_mm"]
    ].mean()
    X_mean["rainfall_mm"] = df.groupby("pincode")["rainfall_2015_mm"].mean()
    X_mean = X_mean[FEATURES]
    z = (X_mean - scaler.mean_) / scaler.scale_
    contrib = z * clf.coef_[0]

    out = {}
    for pincode, row in contrib.iterrows():
        top = row.sort_values(ascending=False).head(3)
        phrases = []
        for feat, c in top.items():
            if c <= 0.05:  # only meaningfully risk-raising factors
                continue
            direction = 1 if z.loc[pincode, feat] > 0 else -1
            phrase = DRIVER_PHRASES.get((feat, direction))
            if phrase:
                phrases.append(phrase)
        out[pincode] = "; ".join(phrases) if phrases else "no dominant driver"
    return pin_df.index.map(out)


def run_financial(cfg: dict, df: pd.DataFrame, models: dict) -> gpd.GeoDataFrame:
    """Cells -> losses -> pincode aggregation -> outputs/pincode_risk.geojson."""
    outputs = Path(cfg["paths"]["outputs"])
    raw = Path(cfg["paths"]["raw"])
    rps = cfg["scenarios"]["return_periods"]
    max_rp = max(rps)

    df = _scenario_losses(cfg, df, models)
    for scen_key in cfg["climate"]["scenarios"]:
        df[f"aal_{scen_key}"] = _aal(cfg, df, scen_key)

    # ---- aggregate cells -> pincodes --------------------------------------
    def wmean(values: str):
        """Exposure-weighted mean: a flooded park is not a flooded hospital."""
        def f(g: pd.DataFrame) -> float:
            w = g["exposure_value_inr"]
            return float(np.average(g[values], weights=w)) if w.sum() > 0 else float(g[values].mean())
        return f

    groups = df.groupby("pincode")
    pin = pd.DataFrame(
        {
            "name": groups["pincode_name"].first(),
            "n_cells": groups.size(),
            "exposure_inr": groups["exposure_value_inr"].sum(),
        }
    )
    for scen_key in cfg["climate"]["scenarios"]:
        pin[f"aal_inr_{scen_key}"] = groups[f"aal_{scen_key}"].sum()
        pin[f"p_flood_rp{max_rp}_{scen_key}"] = groups.apply(
            wmean(f"p_flood_rp{max_rp}_{scen_key}"), include_groups=False
        )
        # The headline 0-100 score (see module docstring for interpretation).
        pin[f"hazard_score_{scen_key}"] = (
            100.0 * pin[f"p_flood_rp{max_rp}_{scen_key}"]
        ).round(1)
    for rp in rps:  # per-RP expected loss, baseline climate (dashboard chart)
        pin[f"loss_rp{rp}_baseline"] = groups[f"loss_rp{rp}_baseline"].sum()
    pin["loss_ratio_baseline"] = pin["aal_inr_baseline"] / pin["exposure_inr"]
    pin["top_drivers"] = _top_drivers(cfg, df, models, pin)

    # ---- attach (approximate!) pincode polygons ----------------------------
    real_bounds = raw / "real" / "pincode_boundaries.geojson"
    bounds_path = real_bounds if real_bounds.exists() else raw / "pincode_boundaries_approx.geojson"
    gdf = gpd.read_file(bounds_path)[["pincode", "geometry"]]
    pin = gdf.merge(pin, on="pincode", how="inner")
    pin["boundary_note"] = (
        "official-source polygons" if real_bounds.exists()
        else "APPROXIMATE Voronoi boundaries — India publishes no official pincode shapes"
    )

    df.to_csv(Path(cfg["paths"]["processed"]) / "cells_results.csv", index=False)
    print(f"  [financial] city AAL (baseline): Rs {pin['aal_inr_baseline'].sum() / 1e7:,.1f} crore")
    return gpd.GeoDataFrame(pin, crs="EPSG:4326")


def save_outputs(cfg: dict, pin: gpd.GeoDataFrame) -> None:
    outputs = Path(cfg["paths"]["outputs"])
    pin.to_file(outputs / "pincode_risk.geojson", driver="GeoJSON")
    pin.drop(columns="geometry").to_csv(outputs / "pincode_risk.csv", index=False)
    print(f"  [financial] wrote {outputs / 'pincode_risk.geojson'} "
          f"({len(pin)} pincodes)")
