"""Data acquisition layer.

real.py      — how to obtain each REAL dataset (URLs, GEE snippets, expected
               filenames) + a loader that resamples user-supplied GeoTIFFs
               onto the analysis grid.
synthetic.py — clearly-labeled SYNTHETIC stand-ins with the same shapes,
               so the pipeline runs end-to-end before any download.

Swap rule: drop the real file at the documented path under data/raw/real/,
re-run `python run_pipeline.py data`, and the real layer replaces the
synthetic one automatically (provenance updates itself).
"""
