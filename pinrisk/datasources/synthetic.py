"""SYNTHETIC sample data generator — clearly-labeled stand-ins for real layers.

============================  READ THIS FIRST  ==============================
Everything produced here is FAKE. It is shaped like real Chennai (a coastal
plain rising gently inland, two east-flowing rivers, a north-south canal,
dense urban core, monsoon rain heavier in the south — echoing Dec 2015) so
the pipeline behaves realistically, but NO output computed from it is a real
risk estimate. Every layer is registered with is_synthetic=True and the
dashboard shows a warning banner until real data replaces it.

Why fake data is *physically coherent* rather than random: the hazard model
must find real structure (low HAND -> flooding) for the plumbing, the spatial
CV, and the validation harness to be exercised honestly.

THE CIRCULARITY CAVEAT (important): the synthetic "2015 flood extent" is
generated from (mostly) the same terrain layers the model trains on, plus
hidden noise fields it cannot see. Validation metrics on synthetic data
therefore prove THE PIPELINE WORKS — they say NOTHING about real-world
accuracy. Real accuracy exists only once real data is in. The hidden fields
(think: clogged drains, micro-topography) keep the scores from being a
perfect 1.0, but the optimism bias remains.
=============================================================================
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from ..grid import cell_area_m2, cell_size_m, grid_shape, project_xy
from ..provenance import LayerProvenance, ProvenanceRegistry
from .real import REAL_SOURCES

# ---------------------------------------------------------------------------
# Approximate centroids of real Chennai PIN codes (pincode, locality, lon, lat).
# APPROXIMATE by construction: India has no authoritative open pincode
# boundaries, so we place seeds at locality centres and draw Voronoi polygons
# around them ("every point belongs to its nearest post office"). Real
# community-derived polygons slot in via data/raw/real/pincode_boundaries.geojson.
# ---------------------------------------------------------------------------
PINCODE_SEEDS = [
    ("600001", "George Town", 80.287, 13.093),
    ("600002", "Anna Salai", 80.270, 13.065),
    ("600003", "Park Town", 80.273, 13.081),
    ("600004", "Mylapore", 80.267, 13.034),
    ("600005", "Triplicane", 80.276, 13.056),
    ("600006", "Thousand Lights", 80.255, 13.059),
    ("600008", "Egmore", 80.260, 13.075),
    ("600010", "Kilpauk", 80.242, 13.080),
    ("600011", "Perambur", 80.240, 13.113),
    ("600012", "Perambur Barracks", 80.252, 13.098),
    ("600013", "Royapuram", 80.288, 13.106),
    ("600014", "Royapettah", 80.264, 13.052),
    ("600015", "Saidapet", 80.223, 13.023),
    ("600016", "St. Thomas Mount", 80.198, 13.005),
    ("600017", "T. Nagar", 80.234, 13.042),
    ("600018", "Teynampet", 80.249, 13.045),
    ("600020", "Adyar", 80.255, 13.006),
    ("600021", "Washermanpet", 80.283, 13.117),
    ("600022", "Guindy (Raj Bhavan)", 80.220, 13.006),
    ("600023", "Ayanavaram", 80.233, 13.100),
    ("600024", "Kodambakkam", 80.226, 13.051),
    ("600026", "Vadapalani", 80.212, 13.050),
    ("600027", "Meenambakkam", 80.164, 12.985),
    ("600028", "R.A. Puram", 80.259, 13.028),
    ("600030", "Shenoy Nagar", 80.228, 13.077),
    ("600031", "Chetpet", 80.243, 13.070),
    ("600032", "Guindy Industrial", 80.212, 13.008),
    ("600033", "West Mambalam", 80.221, 13.038),
    ("600034", "Nungambakkam", 80.242, 13.060),
    ("600035", "Nandanam", 80.239, 13.030),
    ("600037", "Mogappair", 80.173, 13.093),
    ("600038", "ICF Colony", 80.221, 13.088),
    ("600039", "Vyasarpadi", 80.257, 13.118),
    ("600040", "Anna Nagar", 80.210, 13.086),
    ("600041", "Thiruvanmiyur", 80.259, 12.983),
    ("600042", "Velachery", 80.220, 12.979),
    ("600043", "Pallavaram", 80.150, 12.968),
    ("600044", "Chromepet", 80.140, 12.951),
    ("600049", "Villivakkam", 80.208, 13.108),
    ("600053", "Ambattur", 80.148, 13.114),
    ("600061", "Nanganallur", 80.192, 12.982),
    ("600078", "K.K. Nagar", 80.199, 13.041),
    ("600083", "Ashok Nagar", 80.211, 13.036),
    ("600087", "Valasaravakkam", 80.174, 13.042),
    ("600090", "Besant Nagar", 80.266, 12.998),
    ("600091", "Madipakkam", 80.198, 12.962),
    ("600092", "Virugambakkam", 80.190, 13.055),
    ("600093", "Saligramam", 80.200, 13.051),
    ("600096", "Perungudi", 80.245, 12.965),
    ("600101", "Anna Nagar West", 80.196, 13.088),
    ("600102", "Anna Nagar East", 80.222, 13.085),
    ("600106", "Arumbakkam", 80.206, 13.072),
    ("600113", "Tharamani", 80.243, 12.986),
    ("600116", "Porur", 80.156, 13.035),
]


# ------------------------------ small helpers ------------------------------

def _smooth_noise(shape, rng, sigma: float) -> np.ndarray:
    """Spatially-smooth random field, std ~= 1. sigma (in cells) sets the
    'blob size' — big sigma = broad regional wobble, small = local texture."""
    field = gaussian_filter(rng.standard_normal(shape), sigma)
    return field / max(field.std(), 1e-9)


def _coast_lon(lat: np.ndarray) -> np.ndarray:
    """Approximate Chennai shoreline: runs NNE, so the coast's longitude
    increases as you go north. East of this line = Bay of Bengal."""
    return 80.272 + 0.19 * (lat - 13.0)


def _zscore(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Standardise over land cells only (sea would poison mean/std)."""
    m, s = a[mask].mean(), a[mask].std()
    return (a - m) / max(s, 1e-9)


def _rivers(cfg: dict) -> list[np.ndarray]:
    """Synthetic stand-ins for the Cooum, the Adyar and the Buckingham Canal
    as polylines of (lon, lat) points. Real swap: HydroRIVERS + OSM waterways."""
    t = np.linspace(0, 1, 600)
    cooum_lon = 80.12 + 0.17 * t
    cooum_lat = 13.075 - 0.012 * t + 0.010 * np.sin(5 * np.pi * t) * (1 - 0.5 * t)
    adyar_lon = 80.12 + 0.165 * t
    adyar_lat = 12.995 + 0.017 * t + 0.012 * np.sin(4 * np.pi * t) * (1 - 0.4 * t)
    canal_lat = 12.94 + 0.21 * t
    canal_lon = _coast_lon(canal_lat) - 0.015 + 0.003 * np.sin(6 * np.pi * t)
    return [
        np.column_stack([cooum_lon, cooum_lat]),
        np.column_stack([adyar_lon, adyar_lat]),
        np.column_stack([canal_lon, canal_lat]),
    ]


def _voronoi_pincodes(cfg: dict):
    """Voronoi polygons around pincode seeds, clipped to the bbox.
    Returns (GeoDataFrame-like records list, geometry list) built with shapely."""
    from shapely.geometry import MultiPoint, Point, box
    from shapely.ops import voronoi_diagram

    g = cfg["grid"]
    bbox = box(g["lon_min"], g["lat_min"], g["lon_max"], g["lat_max"])
    pts = [Point(lon, lat) for _, _, lon, lat in PINCODE_SEEDS]
    diagram = voronoi_diagram(MultiPoint(pts), envelope=bbox.buffer(0.1))
    records = []
    for cell in diagram.geoms:
        clipped = cell.intersection(bbox)
        if clipped.is_empty:
            continue
        # Match the region back to its seed (voronoi_diagram loses order).
        for (pincode, name, lon, lat), pt in zip(PINCODE_SEEDS, pts):
            if cell.contains(pt):
                records.append(
                    {"pincode": pincode, "name": name, "geometry": clipped}
                )
                break
    return records


# ------------------------------ the generator ------------------------------

def generate_all(cfg: dict) -> ProvenanceRegistry:
    """Generate every layer, save to data/raw/, register provenance.

    Layers written (all 2-D float .npy in grid convention, row 0 = south):
      dem, slope handled downstream; here: dem, dist_coast, dist_drainage,
      drain_elev, imperviousness, landuse_code, building_density,
      rainfall_event, flood_extent_2015, is_sea, pincode_idx
    plus pincode_boundaries_approx.geojson and meta.json.
    """
    rng = np.random.default_rng(cfg["random_seed"])
    raw = Path(cfg["paths"]["raw"])
    g = cfg["grid"]
    shape = grid_shape(cfg)
    n_rows, n_cols = shape

    # Cell-centre coordinate rasters.
    lat = g["lat_min"] + (np.arange(n_rows)[:, None] + 0.5) * g["res_deg"]
    lon = g["lon_min"] + (np.arange(n_cols)[None, :] + 0.5) * g["res_deg"]
    lat2d = np.broadcast_to(lat, shape).copy()
    lon2d = np.broadcast_to(lon, shape).copy()
    x2d, y2d = project_xy(lon2d, lat2d, cfg)  # metres from SW corner
    xy = np.column_stack([x2d.ravel(), y2d.ravel()])

    # --- Sea mask & distance to coast ------------------------------------
    is_sea = lon2d > _coast_lon(lat2d)
    land = ~is_sea
    sea_xy = xy[is_sea.ravel()]
    dist_coast = np.zeros(shape)
    if len(sea_xy):
        d, _ = cKDTree(sea_xy).query(xy, k=1)
        dist_coast = d.reshape(shape)

    # --- Rivers / drainage lines -----------------------------------------
    river_pts = np.vstack(_rivers(cfg))
    on_land = river_pts[:, 0] < _coast_lon(river_pts[:, 1])  # clip mouths at sea
    river_pts = river_pts[on_land]
    rx, ry = project_xy(river_pts[:, 0], river_pts[:, 1], cfg)
    river_tree = cKDTree(np.column_stack([rx, ry]))
    dist_drainage, nearest_riv = river_tree.query(xy, k=1)
    dist_drainage = dist_drainage.reshape(shape)

    # --- DEM: coastal plain + noise, with river valleys carved in --------
    # ~1.2 m elevation gain per km inland (Chennai is famously flat), two
    # noise octaves for regional undulation + local texture.
    dem = (
        1.0
        + 0.0012 * dist_coast
        + 2.0 * _smooth_noise(shape, rng, sigma=8)
        + 0.8 * _smooth_noise(shape, rng, sigma=3)
    )
    dem -= 3.5 * np.exp(-dist_drainage / 250.0)  # valleys along rivers
    dem = np.clip(dem, 0.3, None)
    dem[is_sea] = 0.0

    # Elevation of the nearest drainage point (needed for HAND downstream).
    riv_i = np.clip(((river_pts[:, 1] - g["lat_min"]) / g["res_deg"]).astype(int), 0, n_rows - 1)
    riv_j = np.clip(((river_pts[:, 0] - g["lon_min"]) / g["res_deg"]).astype(int), 0, n_cols - 1)
    river_elev = dem[riv_i, riv_j]
    drain_elev = river_elev[nearest_riv].reshape(shape)

    # --- Land use / imperviousness ----------------------------------------
    # Urban intensity decays from two historical centres (old city near the
    # port, T.Nagar commercial core) + noise -> 3 classes.
    cx1, cy1 = project_xy(80.270, 13.082, cfg)
    cx2, cy2 = project_xy(80.233, 13.045, cfg)
    d1 = np.hypot(x2d - cx1, y2d - cy1)
    d2 = np.hypot(x2d - cx2, y2d - cy2)
    intensity = (
        1.10 * np.exp(-d1 / 7000.0)
        + 0.85 * np.exp(-d2 / 6000.0)
        + 0.30 * _smooth_noise(shape, rng, sigma=10)
        + 0.10
    )
    landuse_code = np.zeros(shape, dtype=np.int8)          # 0 = other
    landuse_code[intensity > 0.42] = 1                     # 1 = urban_sparse
    landuse_code[intensity > 0.75] = 2                     # 2 = urban_dense
    imperviousness = np.choose(landuse_code, [0.18, 0.60, 0.92]) + 0.05 * _smooth_noise(
        shape, rng, sigma=2
    )
    imperviousness = np.clip(imperviousness, 0.0, 1.0)
    imperviousness[is_sea] = 1.0

    # --- Building footprint density ---------------------------------------
    density = np.clip(
        0.60 * imperviousness + 0.12 * _smooth_noise(shape, rng, sigma=2) + 0.05,
        0.01,
        0.70,
    )
    density[landuse_code == 0] *= 0.5
    density[is_sea] = 0.0
    building_density = density  # fraction of cell covered by buildings

    # --- Event rainfall (Dec 2015-like) ------------------------------------
    # Mimics a 0.25-degree product: a 2x2 coarse grid bilinearly upsampled —
    # i.e. essentially uniform at city scale, heavier in the south, exactly
    # like IMD gridded data would look. Values ~ event 24h totals (mm).
    coarse_lat = np.array([g["lat_min"], g["lat_max"]])
    coarse_lon = np.array([g["lon_min"], g["lon_max"]])
    coarse = np.array([[400.0, 375.0], [325.0, 305.0]])  # row 0 = south
    interp = RegularGridInterpolator((coarse_lat, coarse_lon), coarse)
    rainfall = interp(np.column_stack([lat2d.ravel(), lon2d.ravel()])).reshape(shape)
    rainfall += 8.0 * _smooth_noise(shape, rng, sigma=15)

    # --- The "observed" 2015 flood extent (validation target) --------------
    # A hidden truth process the model never sees: physically sensible drivers
    # + two hidden noise fields standing in for clogged drains and
    # micro-topography. See the circularity caveat in the module docstring.
    hand_true = np.clip(dem - drain_elev, 0.0, None)
    logit = (
        2.6 * -_zscore(hand_true, land)
        + 1.2 * -_zscore(dem, land)
        + 1.6 * _zscore(rainfall, land)
        + 0.9 * _zscore(imperviousness, land)
        + 0.6 * -_zscore(np.sqrt(dist_drainage), land)
        + 1.3 * _smooth_noise(shape, rng, sigma=12)   # hidden: regional drainage failure
        + 0.8 * _smooth_noise(shape, rng, sigma=4)    # hidden: micro-topography
    )
    threshold = np.quantile(logit[land], 0.85)        # ~15% of land floods
    flood_extent = ((logit > threshold) & land).astype(np.uint8)

    # --- Pincode polygons (Voronoi around real locality centroids) ---------
    records = _voronoi_pincodes(cfg)
    seed_lon = np.array([s[2] for s in PINCODE_SEEDS])
    seed_lat = np.array([s[3] for s in PINCODE_SEEDS])
    sx, sy = project_xy(seed_lon, seed_lat, cfg)
    _, pincode_idx = cKDTree(np.column_stack([sx, sy])).query(xy, k=1)
    pincode_idx = pincode_idx.reshape(shape).astype(np.int16)

    import geopandas as gpd

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf.to_file(raw / "pincode_boundaries_approx.geojson", driver="GeoJSON")

    # --- Save rasters + metadata -------------------------------------------
    arrays = {
        "dem": dem,
        "dist_coast": dist_coast,
        "dist_drainage": dist_drainage,
        "drain_elev": drain_elev,
        "imperviousness": imperviousness,
        "landuse_code": landuse_code,
        "building_density": building_density,
        "rainfall_event": rainfall,
        "flood_extent_2015": flood_extent,
        "is_sea": is_sea.astype(np.uint8),
        "pincode_idx": pincode_idx,
    }
    for name, arr in arrays.items():
        np.save(raw / f"{name}.npy", arr)

    meta = {
        "bbox": [g["lon_min"], g["lat_min"], g["lon_max"], g["lat_max"]],
        "res_deg": g["res_deg"],
        "shape": list(shape),
        "cell_size_m": list(cell_size_m(cfg)),
        "cell_area_m2": cell_area_m2(cfg),
        "pincodes": [
            {"pincode": p, "name": n, "lon": lo, "lat": la}
            for p, n, lo, la in PINCODE_SEEDS
        ],
        "row0": "south",
    }
    (raw / "meta.json").write_text(json.dumps(meta, indent=2))

    # --- Provenance: honest labels + how to swap in the real thing ---------
    registry = ProvenanceRegistry()
    synth_note = "SYNTHETIC sample data — realistic shape, fake values."
    layer_map = {
        "dem": "dem",
        "rainfall_event": "rainfall_event",
        "flood_extent_2015": "flood_extent_2015",
        "imperviousness": "imperviousness",
        "building_area": "building_area",
        "drainage": "drainage",
        "pincode_boundaries": "pincode_boundaries",
    }
    for layer in layer_map:
        real_info = REAL_SOURCES[layer]
        registry.add(
            LayerProvenance(
                layer=layer,
                source=f"SYNTHETIC stand-in for {real_info['source']}",
                native_resolution=real_info["resolution"],
                is_synthetic=True,
                notes=synth_note + " " + real_info["caveats"],
                swap_instructions=real_info["how_to_get"],
            )
        )
    registry.save(raw / "provenance.json")
    return registry
