"""PinRisk — pincode-level flood risk MVP (Chennai urban flood).

The package mirrors the standard 4-module catastrophe-model architecture:

    hazard/         "how likely / how deep is flooding here?"
    exposure.py     "what value is sitting here to be damaged?"
    vulnerability.py"how much damage does a given flood depth cause?"
    financial.py    "hazard x exposure x vulnerability = expected loss (INR)"

plus the infrastructure they all share:

    grid.py         the analysis grid — the data contract between modules
    provenance.py   where every data layer came from (real vs synthetic)
    datasources/    real-dataset loaders + clearly-labeled synthetic fallbacks
    uncertainty.py  per-pincode confidence scoring
    validation.py   the permanent backtest harness (model vs 2015 flood extent)
"""

__version__ = "0.1.0"
