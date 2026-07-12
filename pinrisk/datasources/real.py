"""Real-dataset acquisition: instructions + loaders.

Nothing here downloads automatically — every source needs either a (free)
account, a Google Earth Engine project, or a bulk-download step that is
better done once by hand. What this module gives you instead:

1. REAL_SOURCES — for each layer: where to get it, at what resolution, its
   known India caveats, and the EXACT filename to save it under so the
   pipeline picks it up automatically.
2. load_real_raster() — reads any GeoTIFF you dropped in data/raw/real/ and
   resamples it onto the analysis grid (needs `rasterio`, an optional
   dependency: pip install rasterio).
3. available() / load_layer() — the check-then-load interface the pipeline
   calls; returns None when a layer is absent so synthetic data can fill in.

Design note: each layer is fetched from a DIFFERENT best-in-class source
(that is normal in cat modelling), then everything is fused by resampling
onto the one shared grid defined in config.yaml.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..grid import grid_shape

# ---------------------------------------------------------------------------
# The catalogue. `expected_file` is relative to data/raw/real/.
# GeoTIFFs must cover the config bounding box; any CRS is fine (reprojected
# on load). Resolution finer or coarser than the grid is fine too —
# bilinear/nearest resampling handles both directions.
# ---------------------------------------------------------------------------
REAL_SOURCES: dict[str, dict] = {
    "dem": {
        "source": "FABDEM v1.2 (Forest And Buildings removed Copernicus DEM)",
        "resolution": "~30 m",
        "expected_file": "dem.tif",
        "how_to_get": (
            "https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn — "
            "download tiles N12E080 & N13E080, mosaic + clip to the config bbox "
            "(gdalwarp or rioxarray), save as data/raw/real/dem.tif. "
            "Alternative: Copernicus GLO-30 via Earth Engine "
            "ee.ImageCollection('COPERNICUS/DEM/GLO30')."
        ),
        "caveats": "FABDEM removes buildings — right choice for flood terrain. "
                   "Still ~30 m: sub-street drainage detail is invisible.",
    },
    "rainfall_event": {
        "source": "IMD gridded daily rainfall (0.25 deg) — Dec 2015 event total",
        "resolution": "0.25 deg (~25 km)",
        "expected_file": "rainfall_event.tif",
        "how_to_get": (
            "IMD Pune gridded rainfall (imdpune.gov.in, free registration) — "
            "extract 2015-11-30..2015-12-02, sum to event total, clip, save. "
            "Alternative: GPM IMERG via Earth Engine ee.ImageCollection('NASA/GPM_L3/IMERG_V07')."
        ),
        "caveats": "25 km rainfall over a city that floods street-by-street is "
                   "the pipeline's coarsest input. Treat as near-uniform forcing; "
                   "terrain provides the local variation.",
    },
    "flood_extent_2015": {
        "source": "Sentinel-1 SAR-derived inundation, Chennai Dec 2015",
        "resolution": "10-20 m",
        "expected_file": "flood_extent_2015.tif  (binary: 1=flooded)",
        "how_to_get": (
            "Earth Engine: ee.ImageCollection('COPERNICUS/S1_GRD') VV, "
            "scenes 2015-12-01..2015-12-05 minus dry baseline (Nov 2015); "
            "threshold backscatter drop (~-15 dB) for open water; export GeoTIFF. "
            "Cross-check with the Global Flood Database (global-flood-database.cloudtostreet.ai) "
            "event DFO_4299 for Chennai 2015. This is the VALIDATION TARGET — "
            "quality here matters more than anywhere else."
        ),
        "caveats": "SAR misses shallow street flooding under dense canopy/buildings "
                   "(urban corner reflections) — treat extent as a lower bound.",
    },
    "imperviousness": {
        "source": "ESA WorldCover 2021 (10 m) built-up class -> imperviousness proxy",
        "resolution": "10 m",
        "expected_file": "imperviousness.tif  (0..1)",
        "how_to_get": (
            "Earth Engine ee.ImageCollection('ESA/WorldCover/v200') or AWS "
            "s3://esa-worldcover; map class 50 (built-up) to 0.9, cropland/grass "
            "to 0.2, tree to 0.1, water to 1.0; aggregate (mean) to the grid."
        ),
        "caveats": "'Built-up' says nothing about drainage quality; it is only "
                   "an imperviousness proxy.",
    },
    "building_area": {
        "source": "Microsoft Global Building Footprints + Google Open Buildings",
        "resolution": "vector footprints",
        "expected_file": "building_area.tif  (built m2 per cell)",
        "how_to_get": (
            "MS: github.com/microsoft/GlobalMLBuildingFootprints (India geojsonl). "
            "Google: sites.research.google/gr/open-buildings (S2 cells covering Chennai). "
            "Union the two (MS better in some wards, Google in others), intersect "
            "with the grid, sum footprint m2 per cell, rasterise."
        ),
        "caveats": "No construction type / value attributes exist for India — "
                   "value per m2 stays an assumption in config.yaml either way.",
    },
    "drainage": {
        "source": "HydroSHEDS / HydroRIVERS + OSM waterways (Cooum, Adyar, Buckingham Canal)",
        "resolution": "vector",
        "expected_file": "drainage.geojson  (LineStrings)",
        "how_to_get": (
            "hydrosheds.org HydroRIVERS Asia shapefile, clip to bbox; ADD OSM "
            "waterway=river|canal via Overpass (HydroRIVERS misses small urban "
            "channels). Save merged lines as GeoJSON."
        ),
        "caveats": "Chennai's storm drains are not in any global dataset; "
                   "distance-to-drainage is computed from major channels only.",
    },
    "pincode_boundaries": {
        "source": "Community-derived pincode polygons (data.gov.in / github mirrors)",
        "resolution": "vector polygons",
        "expected_file": "pincode_boundaries.geojson",
        "how_to_get": (
            "No authoritative open pincode shapefile exists for India (India Post "
            "publishes none) — this gap is real and worth stating to customers. "
            "Community options: datameet pincode polygons (github.com/datameet), "
            "data.gov.in derived boundaries. Save polygons with a 'pincode' "
            "property; they replace the Voronoi approximation."
        ),
        "caveats": "ALL open pincode boundaries are approximate. Outputs must say "
                   "'approximate boundaries' regardless of source.",
    },
}


def real_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["raw"]) / "real"


def available(cfg: dict, layer: str) -> bool:
    """Has the user dropped the real file for this layer in data/raw/real/?"""
    expected = REAL_SOURCES[layer]["expected_file"].split()[0]  # strip comment
    return (real_dir(cfg) / expected).exists()


def load_real_raster(path: Path, cfg: dict, resampling: str = "bilinear") -> np.ndarray:
    """Read a GeoTIFF and resample it onto the analysis grid.

    Requires rasterio (optional dep). Returns a (n_rows, n_cols) array in the
    grid's row convention (row 0 = SOUTH edge — note the flip: GeoTIFFs store
    row 0 at the top/north).
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import reproject
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Reading real GeoTIFFs needs rasterio: .venv/bin/pip install rasterio"
        ) from e

    g = cfg["grid"]
    n_rows, n_cols = grid_shape(cfg)
    # North-up affine transform for the target grid (GeoTIFF convention).
    from rasterio.transform import from_bounds

    dst_transform = from_bounds(
        g["lon_min"], g["lat_min"], g["lon_max"], g["lat_max"], n_cols, n_rows
    )
    dst = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling[resampling],
        )
    return np.flipud(dst)  # GeoTIFF row 0 = north  ->  our row 0 = south


def load_layer(cfg: dict, layer: str) -> np.ndarray | None:
    """Pipeline entry point: real array if the file exists, else None."""
    if layer not in REAL_SOURCES or not available(cfg, layer):
        return None
    expected = REAL_SOURCES[layer]["expected_file"].split()[0]
    path = real_dir(cfg) / expected
    if path.suffix == ".tif":
        resampling = "nearest" if layer == "flood_extent_2015" else "bilinear"
        return load_real_raster(path, cfg, resampling)
    # Vector layers (drainage, pincode boundaries) are handled by the
    # feature-engineering step directly — see hazard/features.py.
    return None
