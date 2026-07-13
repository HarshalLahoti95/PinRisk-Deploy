"""PinRisk dashboard — Streamlit app over the pipeline's output files.

Run:  .venv/bin/streamlit run dashboard.py
(Only reads outputs/ + data/raw/provenance.json — run the pipeline first.)

Design rule: the dashboard NEVER computes risk. It renders numbers the
pipeline wrote to disk, so what you demo is exactly what was validated.
The one exception is the clearly-labelled ILLUSTRATIVE climate toggle,
which switches between columns the pipeline precomputed per scenario.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import folium
import geopandas as gpd
from branca.colormap import LinearColormap
import pandas as pd
import streamlit as st

from pinrisk.config import load_config

st.set_page_config(page_title="PinRisk — Chennai Flood Risk", layout="wide")

CFG = load_config()
OUT = Path(CFG["paths"]["outputs"])

REQUIRED_OUTPUTS = [
    OUT / "pincode_risk.geojson",
    OUT / "validation" / "metrics.json",
    OUT / "validation" / "pincode_validation.csv",
    Path(CFG["paths"]["raw"]) / "provenance.json",
]


def ensure_pipeline_outputs() -> None:
    """Run the pipeline once, on first load, if any required output is missing."""
    if all(p.exists() for p in REQUIRED_OUTPUTS):
        return
    with st.spinner(
        "No pipeline outputs found — running `python run_pipeline.py all` "
        "(first run only, this can take a couple of minutes)..."
    ):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "run_pipeline.py"), "all"],
            capture_output=True, text=True,
        )
    if result.returncode != 0:
        st.error("Pipeline run failed — see the log below.")
        st.code((result.stdout or "") + "\n" + (result.stderr or ""))
        st.stop()


# ---------------------------------------------------------------------------
# Data loading (cached so the map doesn't recompute on every interaction)
# ---------------------------------------------------------------------------
@st.cache_data
def load_outputs():
    pin = gpd.read_file(OUT / "pincode_risk.geojson")
    provenance = json.loads(
        (Path(CFG["paths"]["raw"]) / "provenance.json").read_text()
    )
    metrics = json.loads((OUT / "validation" / "metrics.json").read_text())
    pincode_val = pd.read_csv(OUT / "validation" / "pincode_validation.csv",
                              dtype={"pincode": str})
    return pin, provenance, metrics, pincode_val


def inr(x: float) -> str:
    """Format INR the way an Indian insurer reads it (lakh / crore)."""
    if x >= 1e7:
        return f"Rs {x / 1e7:,.1f} Cr"
    if x >= 1e5:
        return f"Rs {x / 1e5:,.1f} L"
    return f"Rs {x:,.0f}"


ensure_pipeline_outputs()

pin, provenance, metrics, pincode_val = load_outputs()
any_synth = any(v["is_synthetic"] for v in provenance.values())

# ---------------------------------------------------------------------------
# Header + the honesty banner
# ---------------------------------------------------------------------------
st.title("PinRisk — Pincode-level Urban Flood Risk, Chennai")
if any_synth:
    st.error(
        "**SYNTHETIC SAMPLE DATA IN USE** — every number below demonstrates the "
        "pipeline, not real risk. Synthetic layers: "
        + ", ".join(k for k, v in provenance.items() if v["is_synthetic"])
        + ". See README for how to swap in real datasets."
    )

# ---------------------------------------------------------------------------
# Climate scenario toggle (ILLUSTRATIVE — precomputed columns, delta-change)
# ---------------------------------------------------------------------------
scen_options = {v["label"]: k for k, v in CFG["climate"]["scenarios"].items()}
scen_label = st.radio("Climate scenario", list(scen_options), horizontal=True)
scen = scen_options[scen_label]
if scen != "baseline":
    mult = CFG["climate"]["scenarios"][scen]["rain_multiplier"]
    st.warning(
        f"**Illustrative scenario**: rainfall x{mult} via the delta-change method "
        "(same trained model, shifted rainfall). Multipliers are placeholders — "
        "production values must come from NEX-GDDP-CMIP6 / CORDEX-SA, presented "
        "as ranges, not points."
    )

score_col = f"hazard_score_{scen}"
aal_col = f"aal_inr_{scen}"

tab_map, tab_detail, tab_valid, tab_data = st.tabs(
    ["Risk map", "Pincode detail", "Validation (backtest)", "Data & assumptions"]
)

# ---------------------------------------------------------------------------
# Tab 1 — choropleth map of hazard score
# ---------------------------------------------------------------------------
with tab_map:
    c1, c2, c3 = st.columns(3)
    c1.metric("City AAL (expected annual loss)", inr(pin[aal_col].sum()))
    c2.metric("Modelled exposure", inr(pin["exposure_inr"].sum()))
    c3.metric("Highest-risk pincode",
              f"{pin.loc[pin[score_col].idxmax(), 'pincode']} "
              f"({pin.loc[pin[score_col].idxmax(), 'name']})")

    m = folium.Map(location=[13.045, 80.22], zoom_start=12, tiles="cartodbpositron")
    cmap = LinearColormap(
        ["#ffffcc", "#fd8d3c", "#bd0026"],
        vmin=float(pin[score_col].min()), vmax=float(pin[score_col].max()),
        caption="Hazard score (0-100): exposure-weighted % chance of flooding "
                "in the 100-year event",
    )
    folium.GeoJson(
        pin.to_json(),
        style_function=lambda feat: {
            "fillColor": cmap(feat["properties"][score_col]),
            "fillOpacity": 0.75, "color": "#555", "weight": 0.8,
        },
        highlight_function=lambda feat: {"weight": 2.5, "color": "#000"},
        tooltip=folium.GeoJsonTooltip(
            fields=["pincode", "name", score_col, "confidence_label"],
            aliases=["PIN", "Locality", "Hazard score", "Confidence"],
        ),
    ).add_to(m)
    cmap.add_to(m)
    st.iframe(m._repr_html_(), height=560)  # folium renders itself as HTML
    st.caption(f"Boundaries: {pin['boundary_note'].iloc[0]}.")

# ---------------------------------------------------------------------------
# Tab 2 — per-pincode drill-down
# ---------------------------------------------------------------------------
with tab_detail:
    pin_sorted = pin.sort_values(score_col, ascending=False)
    choice = st.selectbox(
        "Pincode (sorted by hazard score)",
        pin_sorted.apply(lambda r: f"{r['pincode']} — {r['name']}", axis=1),
    )
    row = pin_sorted.iloc[
        list(pin_sorted.apply(lambda r: f"{r['pincode']} — {r['name']}", axis=1)).index(choice)
    ]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hazard score", f"{row[score_col]:.0f} / 100")
    c2.metric("Expected annual loss", inr(row[aal_col]))
    c3.metric("Exposure", inr(row["exposure_inr"]))
    c4.metric("Confidence", row["confidence_label"],
              help="Model-agreement score, capped at 0.5 while any data layer "
                   "is synthetic. See README.")

    st.markdown(f"**Top risk drivers:** {row['top_drivers']}")
    st.markdown(f"**Loss ratio (AAL / exposure, baseline):** "
                f"{row['loss_ratio_baseline']:.4%}")

    rps = CFG["scenarios"]["return_periods"]
    loss_curve = pd.DataFrame({
        "return period": [f"1-in-{rp}y" for rp in rps],
        "expected loss (INR)": [row[f"loss_rp{rp}_baseline"] for rp in rps],
    }).set_index("return period")
    st.bar_chart(loss_curve)
    st.caption("Expected loss by rainfall return period (baseline climate). "
               "AAL integrates this curve over annual exceedance probability.")

    with st.expander("Data sources behind this number"):
        st.write(row["data_sources"])
        st.write(f"Boundary: {row['boundary_note']}")

# ---------------------------------------------------------------------------
# Tab 3 — validation / the backtest proof
# ---------------------------------------------------------------------------
with tab_valid:
    if metrics.get("disclaimer", "").startswith("SYNTHETIC"):
        st.error(metrics["disclaimer"])
    st.markdown("**Backtest: model vs the December 2015 Chennai flood extent.** "
                "All model metrics are out-of-fold from spatially-blocked CV — "
                "each cell was scored by a model that never saw its ~2 km "
                "neighbourhood.")

    mrows = []
    for name, s in metrics["models"].items():
        mrows.append({"model": name, "ROC-AUC": round(s["roc_auc"], 3),
                      "PR-AUC": round(s["pr_auc"], 3),
                      "Brier": round(s["brier"], 4) if "brier" in s else None})
    st.dataframe(pd.DataFrame(mrows).set_index("model"))
    st.caption("`trivial_hand_baseline` ranks cells by height-above-drainage "
               "with NO machine learning — the ML must beat it to justify itself.")

    thr = metrics["threshold"]
    st.markdown(
        f"At the prevalence-matched threshold: precision **{thr['precision']}**, "
        f"recall **{thr['recall']}**, CSI **{thr['csi']}** "
        f"(hits {thr['confusion']['tp']:,} / false alarms {thr['confusion']['fp']:,} "
        f"/ misses {thr['confusion']['fn']:,})"
    )
    hr = metrics["pincode_hit_rate"]
    st.markdown(f"**Pincode hit rate:** {hr['n_hit']}/{hr['n_flooded_pincodes']} "
                f"materially-flooded pincodes correctly flagged "
                f"({hr['hit_rate']:.0%}).")

    img = OUT / "validation" / "predicted_vs_actual.png"
    if img.exists():
        st.image(str(img), width="stretch")
    st.dataframe(pincode_val.sort_values("actual_flooded_frac", ascending=False),
                 height=300)

# ---------------------------------------------------------------------------
# Tab 4 — provenance & assumptions (the credibility tab)
# ---------------------------------------------------------------------------
with tab_data:
    st.markdown("### Data provenance")
    prov_df = pd.DataFrame(provenance).T[
        ["source", "native_resolution", "is_synthetic", "notes"]
    ]
    st.dataframe(prov_df, height=300)

    st.markdown("### Key assumptions (all editable in `config.yaml`)")
    e = CFG["exposure"]
    st.markdown(
        f"- **Asset values**: dense urban Rs {e['value_inr_per_m2']['urban_dense']:,}/m2, "
        f"sparse Rs {e['value_inr_per_m2']['urban_sparse']:,}/m2, other "
        f"Rs {e['value_inr_per_m2']['other']:,}/m2; contents factor x{e['contents_factor']}.\n"
        f"- **Depth-damage curve**: JRC-style (Huizinga et al. 2017, Asia "
        f"residential, approximate digitisation).\n"
        f"- **Depth-if-flooded**: heuristic `ref_depth(RP) * exp(-HAND/"
        f"{CFG['vulnerability']['hand_decay_m']} m)` — placeholder for a "
        f"hydraulic model.\n"
        f"- **Rainfall frequency**: 2015 event treated as ~1-in-100y; RP scaling "
        f"factors are placeholders for IMD frequency analysis.\n"
        f"- **AAL integration**: trapezoid over 3 return periods; zero loss below "
        f"RP10; flat tail beyond RP100."
    )
    curve = CFG["vulnerability"]["depth_damage_curve"]
    st.line_chart(pd.DataFrame(
        {"damage fraction": list(curve.values())},
        index=pd.Index(list(curve.keys()), name="flood depth (m)"),
    ))
