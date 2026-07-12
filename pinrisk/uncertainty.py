"""Uncertainty & confidence — every pincode output carries "how sure are we?"

MVP confidence signal (deliberately simple, honestly labelled):

1. MODEL DISAGREEMENT. We trained two different model families on the same
   data. Where the logistic and GBM probabilities agree, structure in the
   data supports the prediction; where they diverge, it is model-choice
   artefact -> lower confidence. (Cheap stand-in for proper conformal
   prediction / quantile intervals — the named post-MVP upgrade.)

2. DATA QUALITY CAP. If any core layer is synthetic, confidence is capped at
   0.5 — the model literally cannot be more than "demo confident" on fake
   data. Real layers lift the cap automatically via provenance.

Confidence label: >= 0.60 High | >= 0.35 Medium | else Low.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd

from .provenance import ProvenanceRegistry


def add_confidence(
    cfg: dict, cells: pd.DataFrame, pin: gpd.GeoDataFrame, registry: ProvenanceRegistry
) -> gpd.GeoDataFrame:
    """Append confidence_score / confidence_label / data_sources per pincode."""
    cells = cells.copy()
    cells["disagreement"] = (cells["p_rp100_logistic"] - cells["p_rp100_gbm"]).abs()

    def wmean(g: pd.DataFrame) -> float:
        w = g["exposure_value_inr"]
        return float(np.average(g["disagreement"], weights=w)) if w.sum() > 0 else float(
            g["disagreement"].mean()
        )

    dis = cells.groupby("pincode").apply(wmean, include_groups=False)
    # Map disagreement -> [0, 1]: 0 disagreement = 1.0; >= 0.25 absolute
    # probability gap = 0. The 0.25 scale is a judgment call, stated here.
    agreement_conf = (1.0 - dis / 0.25).clip(0.0, 1.0)

    quality_cap = 0.5 if registry.any_synthetic() else 1.0
    conf = (agreement_conf * quality_cap).rename("confidence_score")

    pin = pin.merge(conf, left_on="pincode", right_index=True, how="left")
    pin["confidence_score"] = pin["confidence_score"].fillna(0.0).round(3)
    pin["confidence_label"] = pd.cut(
        pin["confidence_score"],
        bins=[-0.01, 0.35, 0.60, 1.01],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    pin["data_sources"] = registry.sources_summary()
    if registry.any_synthetic():
        print(f"  [uncertainty] confidence capped at 0.5 — synthetic layers: "
              f"{', '.join(registry.synthetic_layers())}")
    return pin
