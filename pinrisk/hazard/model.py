"""Flood-susceptibility model — the heart of the hazard module.

WHAT IT PREDICTS
    P(cell floods | an event with the given rainfall occurs)
    — a *conditional spatial* probability. The *temporal* probability (how
    often such rainfall happens) enters later via return periods in the
    financial module. Keeping "where" and "how often" separate is standard
    cat-model practice and avoids double counting.

WHY TWO MODELS (both kept, on purpose)
    logistic  — penalised logistic regression on standardised features.
                Every coefficient is a signed, explainable weight ("risk
                falls with HAND"). The honesty floor; hard to overfit.
    gbm       — HistGradientBoosting (sklearn's LightGBM-style trees).
                Learns nonlinearities & interactions (imperviousness x
                rainfall). The accuracy ceiling for tabular data at this
                scale. Kept honest by MONOTONIC CONSTRAINTS (below).
    ensemble  — their mean. Their DISAGREEMENT doubles as a per-location
                uncertainty signal (used by uncertainty.py).

    Deliberately NOT deep learning: with ~15% positive labels from a single
    event, a CNN would overfit, be unexplainable to actuaries, and be
    unmaintainable by a non-ML-specialist team. See README "Next steps".

PARAMETER WEIGHTING — answered here, once and for all:
    We never hand-assign weights to elevation vs rainfall. The model LEARNS
    them from the flood inventory; we CONSTRAIN their direction with physics
    (monotonic constraints); we READ them back via standardised coefficients
    / feature importance; and validation tells us if they're right.

VALIDATION DESIGN
    Spatially-blocked GroupKFold (~2 km blocks). Every cell gets an
    out-of-fold (OOF) prediction — made by a model that never saw that
    cell's neighbourhood — and ALL reported metrics use OOF predictions
    only. The final model is refit on all data for scenario prediction,
    which is standard: validate honestly, then use everything you have.

    A TRIVIAL BASELINE (rank cells by -HAND, no ML at all) is always
    reported alongside. If ML can't beat it, the ML is adding noise.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..grid import spatial_blocks

# The predictor set. Order matters (monotonic constraints align by position).
FEATURES = [
    "elevation_m",
    "slope_deg",
    "hand_m",
    "dist_drainage_m",
    "dist_coast_m",
    "imperviousness",
    "rainfall_mm",
]

# Physics as constraints, not hand-picked weights:
#   -1 risk must not increase with the feature | +1 must not decrease | 0 free
MONOTONE = {
    "elevation_m": -1,      # higher ground -> never more risk
    "slope_deg": 0,         # ambiguous (sheds water but also receives runoff)
    "hand_m": -1,           # higher above drainage -> never more risk
    "dist_drainage_m": -1,  # farther from channels -> never more fluvial risk
    "dist_coast_m": 0,      # left free (surge vs inland ponding trade-off)
    "imperviousness": 1,    # more pavement -> never less risk
    "rainfall_mm": 1,       # more rain -> never less risk
}


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs. The rainfall column is scenario-dependent, so it is
    named generically 'rainfall_mm' — training uses the 2015 event values,
    scenario prediction swaps in scaled values (delta-change pattern)."""
    X = df[[f for f in FEATURES if f != "rainfall_mm"]].copy()
    X["rainfall_mm"] = df["rainfall_2015_mm"]
    return X[FEATURES]


def _make_models(cfg: dict) -> dict:
    return {
        "logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",  # rare positives: reweight, don't ignore
                        C=1.0,
                        max_iter=2000,
                    ),
                ),
            ]
        ),
        "gbm": HistGradientBoostingClassifier(
            class_weight="balanced",
            monotonic_cst=[MONOTONE[f] for f in FEATURES],
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=50,     # each leaf must cover ~a city block: anti-overfit
            l2_regularization=1.0,
            random_state=cfg["random_seed"],
        ),
    }


def train_and_validate(cfg: dict, df: pd.DataFrame) -> dict:
    """Spatially-blocked CV -> OOF predictions -> metrics; then refit on all.

    Writes:
      processed/hazard_oof.csv      per-cell OOF probabilities (validation fuel)
      processed/hazard_models.pkl   final models refit on all data
      processed/hazard_metrics.json CV metrics incl. trivial baseline
    """
    processed = Path(cfg["paths"]["processed"])
    X = _feature_matrix(df)
    y = df["flooded_2015"].to_numpy()
    groups = spatial_blocks(df, cfg["hazard_model"]["block_cells"])
    n_folds = cfg["hazard_model"]["n_cv_folds"]

    oof = pd.DataFrame({"cell_id": df["cell_id"], "fold": -1})
    fold_auc: dict[str, list] = {"logistic": [], "gbm": []}
    cv = GroupKFold(n_splits=n_folds)
    for k, (tr, te) in enumerate(cv.split(X, y, groups)):
        for name, model in _make_models(cfg).items():
            model.fit(X.iloc[tr], y[tr])
            p = model.predict_proba(X.iloc[te])[:, 1]
            oof.loc[oof.index[te], f"p_{name}"] = p
            fold_auc[name].append(roc_auc_score(y[te], p))
        oof.loc[oof.index[te], "fold"] = k
    oof["p_ensemble"] = 0.5 * (oof["p_logistic"] + oof["p_gbm"])
    oof["flooded_2015"] = y

    # ---- metrics: every model vs the no-ML trivial baseline ---------------
    def _scores(scores: np.ndarray, proper_probability: bool) -> dict:
        out = {
            "roc_auc": float(roc_auc_score(y, scores)),
            "pr_auc": float(average_precision_score(y, scores)),
        }
        if proper_probability:
            out["brier"] = float(brier_score_loss(y, scores))
        return out

    metrics = {
        "prevalence": float(y.mean()),
        "n_cells": int(len(y)),
        "cv": {
            "folds": n_folds,
            "block_cells": cfg["hazard_model"]["block_cells"],
            "fold_roc_auc": {k: [float(v) for v in vs] for k, vs in fold_auc.items()},
        },
        "models": {
            "trivial_hand_baseline": _scores(-df["hand_m"].to_numpy(), False),
            "logistic": _scores(oof["p_logistic"].to_numpy(), True),
            "gbm": _scores(oof["p_gbm"].to_numpy(), True),
            "ensemble": _scores(oof["p_ensemble"].to_numpy(), True),
        },
    }

    # ---- learned weights, readable ----------------------------------------
    # Standardised logistic coefficients ARE the answer to "what weight does
    # elevation get vs rainfall": learned, signed, comparable across features.
    final_models = {name: m.fit(X, y) for name, m in _make_models(cfg).items()}
    coefs = final_models["logistic"].named_steps["clf"].coef_[0]
    metrics["learned_weights_logistic"] = {
        f: round(float(c), 3) for f, c in zip(FEATURES, coefs)
    }

    oof.to_csv(processed / "hazard_oof.csv", index=False)
    with open(processed / "hazard_models.pkl", "wb") as f:
        pickle.dump(final_models, f)
    (processed / "hazard_metrics.json").write_text(json.dumps(metrics, indent=2))

    ens, triv = metrics["models"]["ensemble"], metrics["models"]["trivial_hand_baseline"]
    print(f"  [hazard] spatial-CV ROC-AUC  ensemble={ens['roc_auc']:.3f}  "
          f"trivial(-HAND)={triv['roc_auc']:.3f}")
    print(f"  [hazard] spatial-CV PR-AUC   ensemble={ens['pr_auc']:.3f}  "
          f"trivial(-HAND)={triv['pr_auc']:.3f}  (prevalence={y.mean():.2f})")
    print(f"  [hazard] learned weights (std. logistic coefs): "
          f"{metrics['learned_weights_logistic']}")
    if ens["roc_auc"] <= triv["roc_auc"]:
        print("  [hazard] WARNING: ML does not beat the trivial HAND baseline — "
              "the ML is adding nothing. Investigate before trusting outputs.")
    return metrics


def load_models(cfg: dict) -> dict:
    with open(Path(cfg["paths"]["processed"]) / "hazard_models.pkl", "rb") as f:
        return pickle.load(f)


def predict(models: dict, df: pd.DataFrame, rainfall_mm: np.ndarray) -> dict:
    """Predict flood probability under a given rainfall field.

    This is the delta-change lever: return periods and climate scenarios are
    all expressed as scaled rainfall pushed through the SAME trained model.
    Returns {'logistic': p, 'gbm': p, 'ensemble': p} aligned with df rows.
    """
    X = _feature_matrix(df)
    X["rainfall_mm"] = rainfall_mm
    out = {name: m.predict_proba(X)[:, 1] for name, m in models.items()}
    out["ensemble"] = 0.5 * (out["logistic"] + out["gbm"])
    return out
