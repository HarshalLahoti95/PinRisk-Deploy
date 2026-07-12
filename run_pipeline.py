#!/usr/bin/env python
"""PinRisk pipeline orchestrator — runs the cat-model assembly line in order.

    python run_pipeline.py all         # everything, in order (first run: do this)
    python run_pipeline.py data        # acquire layers (real if present, else synthetic)
    python run_pipeline.py features    # rasters -> the per-cell feature table
    python run_pipeline.py hazard      # train + spatially validate the susceptibility model
    python run_pipeline.py financial   # exposure -> vulnerability -> INR losses -> pincodes
    python run_pipeline.py validate    # backtest vs the 2015 flood extent

Stages communicate ONLY through files (data/processed/*, outputs/*), so each
can be re-run alone after you change its inputs — e.g. drop in a real DEM,
then re-run from `data` onward. Stage grouping is about workflow; the four
cat-model modules stay separate Python files regardless.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pinrisk.config import ensure_dirs, load_config
from pinrisk.provenance import ProvenanceRegistry


def stage_data(cfg: dict) -> None:
    """Acquire every layer. Real files in data/raw/real/ win; synthetic
    stand-ins (clearly labelled in provenance) fill every gap."""
    from pinrisk.datasources import real, synthetic

    print("[1/5] data acquisition")
    have_real = [l for l in real.REAL_SOURCES if real.available(cfg, l)]
    if have_real:
        print(f"  found real files for: {', '.join(have_real)}")
    else:
        print("  no real files found in data/raw/real/ — generating synthetic "
              "sample data for ALL layers (see README for swap instructions)")
    registry = synthetic.generate_all(cfg)
    # Layers with a real file override the synthetic provenance entry.
    for layer in have_real:
        info = real.REAL_SOURCES[layer]
        from pinrisk.provenance import LayerProvenance

        registry.add(LayerProvenance(
            layer=layer, source=info["source"],
            native_resolution=info["resolution"], is_synthetic=False,
            notes=info["caveats"], swap_instructions="",
        ))
    registry.save(Path(cfg["paths"]["raw"]) / "provenance.json")
    if registry.any_synthetic():
        print(f"  SYNTHETIC layers in use: {', '.join(registry.synthetic_layers())}")


def stage_features(cfg: dict) -> None:
    from pinrisk.hazard.features import build_features

    print("[2/5] feature engineering")
    build_features(cfg)


def stage_hazard(cfg: dict) -> None:
    import pandas as pd
    from pinrisk.hazard.model import train_and_validate

    print("[3/5] hazard model (train + spatially-blocked CV)")
    # dtype note: pincode must stay a string ("600001"), else pandas parses
    # it as an integer and merges with GeoJSON keys fail.
    df = pd.read_csv(Path(cfg["paths"]["processed"]) / "features.csv",
                     dtype={"pincode": str})
    train_and_validate(cfg, df)


def stage_financial(cfg: dict) -> None:
    import pandas as pd
    from pinrisk.exposure import compute_exposure
    from pinrisk.financial import run_financial, save_outputs
    from pinrisk.hazard.model import load_models
    from pinrisk.uncertainty import add_confidence

    print("[4/5] exposure -> vulnerability -> financial -> pincode aggregation")
    df = pd.read_csv(Path(cfg["paths"]["processed"]) / "features.csv",
                     dtype={"pincode": str})
    df = compute_exposure(cfg, df)
    models = load_models(cfg)
    pin = run_financial(cfg, df, models)
    registry = ProvenanceRegistry.load(Path(cfg["paths"]["raw"]) / "provenance.json")
    cells = pd.read_csv(Path(cfg["paths"]["processed"]) / "cells_results.csv",
                        dtype={"pincode": str})
    pin = add_confidence(cfg, cells, pin, registry)
    save_outputs(cfg, pin)


def stage_validate(cfg: dict) -> None:
    from pinrisk.validation import run_validation

    print("[5/5] validation — backtest vs 2015 flood extent")
    run_validation(cfg)


STAGES = {
    "data": stage_data,
    "features": stage_features,
    "hazard": stage_hazard,
    "financial": stage_financial,
    "validate": stage_validate,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=[*STAGES, "all"],
                        help="pipeline stage to run ('all' runs every stage in order)")
    parser.add_argument("--config", default=None, help="alternative config.yaml path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    todo = list(STAGES.values()) if args.stage == "all" else [STAGES[args.stage]]
    t0 = time.time()
    for fn in todo:
        fn(cfg)
    print(f"\nDone in {time.time() - t0:.1f}s.")
    if args.stage == "all":
        print("Next: .venv/bin/streamlit run dashboard.py")


if __name__ == "__main__":
    sys.exit(main())
