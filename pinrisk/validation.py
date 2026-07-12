"""VALIDATION harness — the backtest against the 2015 flood extent.

This is permanent infrastructure, not a one-off check: every future model
change, new city, or new data layer re-runs THIS harness and must not
degrade THESE metrics. It is also the artifact that sells the model
("we predicted the 2015 flood footprint with X on held-out areas").

What makes it honest:
  * metrics use OUT-OF-FOLD predictions only — each cell was scored by a
    model that never trained on its ~2 km neighbourhood (no spatial leakage);
  * the no-ML trivial baseline (-HAND ranking) is always shown alongside;
  * plain accuracy is never reported (15% prevalence makes it meaningless);
  * on synthetic data, a disclaimer is stamped on every figure and metric
    file: pipeline-proof, not real-world skill (see synthetic.py docstring
    on circularity).

Metrics glossary:
  ROC-AUC    P(random flooded cell ranks above random dry cell). 0.5 = coin flip.
  PR-AUC     precision-recall area — the honest metric under class imbalance;
             compare against prevalence (~0.15), not against 1.0.
  Brier      mean squared error of predicted probability (calibration).
  CSI        hits / (hits + misses + false alarms) — the flood-mapping
             community's standard score ("critical success index").
  Pincode hit rate — of pincodes that materially flooded (>10% of area),
             what fraction did the model also flag (>10% predicted)?
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: we save PNGs, never open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix

from .grid import to_raster
from .provenance import ProvenanceRegistry

DISCLAIMER = ("SYNTHETIC SAMPLE DATA — these numbers validate the pipeline "
              "plumbing, NOT real-world accuracy.")


def run_validation(cfg: dict) -> dict:
    """Compare OOF predictions vs the observed 2015 extent. Writes
    outputs/validation/{metrics.json, pincode_validation.csv,
    predicted_vs_actual.png}."""
    processed = Path(cfg["paths"]["processed"])
    outdir = Path(cfg["paths"]["outputs"]) / "validation"
    outdir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(processed / "features.csv", dtype={"pincode": str})
    oof = pd.read_csv(processed / "hazard_oof.csv")
    df = cells.merge(oof.drop(columns=["flooded_2015"]), on="cell_id")
    registry = ProvenanceRegistry.load(Path(cfg["paths"]["raw"]) / "provenance.json")
    hazard_metrics = json.loads((processed / "hazard_metrics.json").read_text())

    y = df["flooded_2015"].to_numpy()
    p = df["p_ensemble"].to_numpy()

    # ---- threshold: prevalence-matched ------------------------------------
    # To draw a binary "predicted flood map" we flag exactly as many cells as
    # actually flooded (top-15% probabilities). Simple, defensible, and free
    # of threshold-shopping; production would optimise CSI on a holdout.
    prevalence = y.mean()
    thr = float(np.quantile(p, 1.0 - prevalence))
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    csi = tp / (tp + fn + fp)

    metrics = {
        "disclaimer": DISCLAIMER if registry.any_synthetic() else "real data",
        "note": "All model metrics computed on OUT-OF-FOLD predictions from "
                "spatially-blocked CV (no cell graded by a model that saw its "
                "neighbourhood).",
        "models": hazard_metrics["models"],           # ROC/PR/Brier incl. trivial
        "threshold": {
            "method": "prevalence-matched",
            "value": round(thr, 4),
            "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
            "precision": round(float(tp / (tp + fp)), 3),
            "recall": round(float(tp / (tp + fn)), 3),
            "csi": round(float(csi), 3),
        },
    }

    # ---- pincode-level hit rate -------------------------------------------
    pin = df.groupby("pincode").agg(
        name=("pincode_name", "first"),
        actual_flooded_frac=("flooded_2015", "mean"),
        predicted_flooded_frac=("p_ensemble", lambda s: float((s >= thr).mean())),
    )
    flooded = pin[pin["actual_flooded_frac"] > 0.10]
    hits = (flooded["predicted_flooded_frac"] > 0.10).sum()
    metrics["pincode_hit_rate"] = {
        "definition": "pincodes with >10% area flooded that the model also "
                      "flags at >10% predicted",
        "n_flooded_pincodes": int(len(flooded)),
        "n_hit": int(hits),
        "hit_rate": round(float(hits / max(len(flooded), 1)), 3),
    }
    pin.round(3).to_csv(outdir / "pincode_validation.csv")

    # ---- the three-panel proof figure --------------------------------------
    g = cfg["grid"]
    extent = [g["lon_min"], g["lon_max"], g["lat_min"], g["lat_max"]]
    prob_r = to_raster(df, "p_ensemble", cfg)
    actual_r = to_raster(df, "flooded_2015", cfg)
    pred_r = to_raster(df.assign(pred=pred), "pred", cfg)
    # Agreement categories: 0 dry-correct, 1 hit, 2 false alarm, 3 miss.
    agree = np.where(
        np.isnan(actual_r), np.nan,
        np.select(
            [(pred_r == 1) & (actual_r == 1), (pred_r == 1) & (actual_r == 0),
             (pred_r == 0) & (actual_r == 1)],
            [1, 2, 3],
            default=0,
        ),
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)
    im0 = axes[0].imshow(prob_r, origin="lower", extent=extent, cmap="YlGnBu",
                         vmin=0, vmax=1)
    axes[0].set_title("Predicted flood probability (out-of-fold)")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    axes[1].imshow(actual_r, origin="lower", extent=extent,
                   cmap=ListedColormap(["#f0f0f0", "#08519c"]), vmin=0, vmax=1)
    axes[1].set_title('"Observed" Dec-2015 flood extent'
                      + (" (SYNTHETIC)" if registry.any_synthetic() else ""))
    axes[2].imshow(agree, origin="lower", extent=extent,
                   cmap=ListedColormap(["#f0f0f0", "#2b8cbe", "#fdae61", "#d7191c"]),
                   vmin=0, vmax=3)
    axes[2].set_title(f"Agreement — hit (blue) / false alarm (orange) / miss (red)\n"
                      f"CSI={csi:.2f}, recall={metrics['threshold']['recall']:.2f}, "
                      f"precision={metrics['threshold']['precision']:.2f}")
    for ax in axes:
        ax.set_xlabel("lon"), ax.set_ylabel("lat")
    if registry.any_synthetic():
        fig.suptitle(DISCLAIMER, color="crimson", fontsize=13, fontweight="bold")
    fig.savefig(outdir / "predicted_vs_actual.png", dpi=130)
    plt.close(fig)

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    ens = metrics["models"]["ensemble"]
    triv = metrics["models"]["trivial_hand_baseline"]
    print(f"  [validate] OOF ROC-AUC {ens['roc_auc']:.3f} (trivial {triv['roc_auc']:.3f}) | "
          f"PR-AUC {ens['pr_auc']:.3f} (trivial {triv['pr_auc']:.3f}, prevalence {prevalence:.2f})")
    print(f"  [validate] CSI {csi:.3f} | pincode hit rate "
          f"{metrics['pincode_hit_rate']['n_hit']}/{metrics['pincode_hit_rate']['n_flooded_pincodes']}")
    print(f"  [validate] wrote {outdir / 'predicted_vs_actual.png'}")
    if registry.any_synthetic():
        print(f"  [validate] {DISCLAIMER}")
    return metrics
