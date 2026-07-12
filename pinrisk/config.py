"""Load config.yaml — the single source of truth for every assumption.

Usage:
    from pinrisk.config import load_config
    cfg = load_config()          # reads config.yaml next to run_pipeline.py
    cfg["grid"]["res_deg"]       # plain nested dicts, no magic
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Project root = the folder containing config.yaml (one level above pinrisk/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """Read the YAML config into a nested dict.

    YAML quirk worth knowing: keys like `10:` parse as *integers*, so the
    return-period lookups (cfg["scenarios"]["rainfall_scale"][100]) use int
    keys — convenient for us, surprising if you expected strings.
    """
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Resolve data paths relative to the project root so the pipeline can be
    # launched from any working directory.
    for key, rel in cfg["paths"].items():
        cfg["paths"][key] = str(PROJECT_ROOT / rel)
    return cfg


def ensure_dirs(cfg: dict) -> None:
    """Create data/output folders if missing (safe to call repeatedly)."""
    for p in cfg["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)
    # Real datasets, when you download them, go here (see datasources/real.py).
    Path(cfg["paths"]["raw"], "real").mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["outputs"], "validation").mkdir(parents=True, exist_ok=True)
