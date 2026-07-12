"""EXPOSURE module — "what value (INR) is sitting in each cell?"

exposure = built-up floor area x assumed replacement value per m2 x a
contents factor. That's it — deliberately simple, with every number coming
from config.yaml (exposure:) so an underwriter can challenge and change it.

Honesty note: no dataset gives building VALUE for India. Footprints are real
(Microsoft/Google), values are ASSUMPTIONS. In production the insurer's own
portfolio (sums insured per address) replaces this whole module's guesswork —
which is exactly why the module boundary exists.
"""

from __future__ import annotations

import pandas as pd


def compute_exposure(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Append `exposure_value_inr` to the cells table."""
    e = cfg["exposure"]
    value_per_m2 = df["landuse"].map(e["value_inr_per_m2"]).astype(float)
    df["exposure_value_inr"] = (
        df["built_area_m2"] * value_per_m2 * e["contents_factor"]
    )
    total_cr = df["exposure_value_inr"].sum() / 1e7  # 1 crore = 10^7 INR
    print(f"  [exposure] total modelled exposure: Rs {total_cr:,.0f} crore "
          f"(assumption-driven — see config.yaml exposure:)")
    return df
