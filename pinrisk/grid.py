"""The analysis grid — THE data contract between all four cat-model modules.

Mental model: a spreadsheet laid over Chennai. Each row of the "cells table"
is one ~110 m x 110 m square of the city; each column is one fact about that
square (elevation, rainfall, flood probability, INR exposure, ...).

Modules never talk to each other directly — they read columns others wrote
and append their own. That is what makes hazard/exposure/vulnerability/
financial swappable independently (different peril, different city, same
skeleton).

Two representations of the same data, and helpers to flip between them:
  * "table"  — pandas DataFrame, one row per cell   (good for ML / groupby)
  * "raster" — 2-D numpy array shaped (n_rows, n_cols) (good for terrain math
               and for drawing maps)

Coordinate conventions used everywhere:
  i = row index, 0 at the SOUTH edge, increases northward (latitude)
  j = col index, 0 at the WEST edge, increases eastward  (longitude)
  cell centres: lat = lat_min + (i + 0.5) * res,  lon = lon_min + (j + 0.5) * res

Distances: at city scale we use a local equirectangular projection
(metres east/north of the grid's SW corner). Good to <0.1% error over 20 km.
A production system should use a proper projected CRS (UTM zone 44N,
EPSG:32644) via pyproj — noted, not needed for the MVP's accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Metres per degree of latitude (nearly constant everywhere on Earth).
M_PER_DEG_LAT = 110_574.0


def m_per_deg_lon(lat_deg: float) -> float:
    """Metres per degree of longitude — shrinks with cos(latitude)."""
    return 111_320.0 * np.cos(np.radians(lat_deg))


def grid_shape(cfg: dict) -> tuple[int, int]:
    """(n_rows, n_cols) of the analysis grid from the config bounding box."""
    g = cfg["grid"]
    n_rows = int(round((g["lat_max"] - g["lat_min"]) / g["res_deg"]))
    n_cols = int(round((g["lon_max"] - g["lon_min"]) / g["res_deg"]))
    return n_rows, n_cols


def cell_size_m(cfg: dict) -> tuple[float, float]:
    """(height_m, width_m) of one cell at the grid's central latitude."""
    g = cfg["grid"]
    lat_mid = 0.5 * (g["lat_min"] + g["lat_max"])
    return g["res_deg"] * M_PER_DEG_LAT, g["res_deg"] * m_per_deg_lon(lat_mid)


def cell_area_m2(cfg: dict) -> float:
    h, w = cell_size_m(cfg)
    return h * w


def base_table(cfg: dict) -> pd.DataFrame:
    """Build the empty cells table: one row per grid cell with ids + coords."""
    g = cfg["grid"]
    n_rows, n_cols = grid_shape(cfg)
    ii, jj = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    i = ii.ravel()
    j = jj.ravel()
    return pd.DataFrame(
        {
            "cell_id": i * n_cols + j,
            "i": i,
            "j": j,
            "lat": g["lat_min"] + (i + 0.5) * g["res_deg"],
            "lon": g["lon_min"] + (j + 0.5) * g["res_deg"],
        }
    )


def project_xy(lon, lat, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Lon/lat -> metres east/north of the grid's SW corner (local flat-earth)."""
    g = cfg["grid"]
    lat_mid = 0.5 * (g["lat_min"] + g["lat_max"])
    x = (np.asarray(lon) - g["lon_min"]) * m_per_deg_lon(lat_mid)
    y = (np.asarray(lat) - g["lat_min"]) * M_PER_DEG_LAT
    return x, y


def to_raster(df: pd.DataFrame, col: str, cfg: dict, fill=np.nan) -> np.ndarray:
    """Table column -> 2-D array for terrain math / plotting.

    Cells missing from `df` (e.g. sea cells dropped from the analysis)
    become `fill`.
    """
    arr = np.full(grid_shape(cfg), fill, dtype=float)
    arr[df["i"].to_numpy(), df["j"].to_numpy()] = df[col].to_numpy(dtype=float)
    return arr


def raster_to_col(df: pd.DataFrame, arr: np.ndarray) -> np.ndarray:
    """2-D array -> values aligned with the rows of `df` (inverse of to_raster)."""
    return arr[df["i"].to_numpy(), df["j"].to_numpy()]


def spatial_blocks(df: pd.DataFrame, block_cells: int) -> np.ndarray:
    """Assign each cell to a square spatial block (used for blocked CV).

    Why: flood status is spatially autocorrelated — neighbouring cells are
    near-copies. A random train/test split would place a cell's neighbour in
    the training set and grade the model on questions it has effectively seen
    ("spatial leakage"), inflating scores. Blocked CV holds out whole ~2 km
    squares instead, which is the honest test of predicting *new places*.
    """
    bi = df["i"].to_numpy() // block_cells
    bj = df["j"].to_numpy() // block_cells
    return bi * 10_000 + bj  # unique id per (bi, bj) block
