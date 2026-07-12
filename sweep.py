"""
sweep.py
Hyperparameter sweep for the XGBoost component of the ensemble.
Runs RandomizedSearchCV, logs all trial results to data/sweep_results.csv,
and saves the best model.

Run: python sweep.py
Run faster: python sweep.py --n_iter 10 --cv 3
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_pipeline import build_dataset
from feature_engineering import MolecularFeatureEngineer
from model import DrugRepurposingModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "compounds.csv")
SWEEP_LOG = os.path.join(os.path.dirname(__file__), "data", "sweep_results.csv")

PARAM_GRID = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [3, 4, 6, 8],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma":            [0, 0.1, 0.3],
}


def run_sweep(n_iter=20, cv_folds=5, random_state=42):
    # ── Load + featurize ──────────────────────────────────────────────────────
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = build_dataset()

    engineer = MolecularFeatureEngineer(ecfp_bits=2048)
    X, y = engineer.fit_transform(df)

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    spw   = round(n_neg / max(n_pos, 1), 2)
    log.info(f"Dataset: {len(X)} compounds | Actives: {n_pos} | scale_pos_weight: {spw}")

    # ── RandomizedSearchCV on XGBoost ─────────────────────────────────────────
    base_xgb = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=spw,
        random_state=random_state,
        verbosity=0,
    )
    cv     = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    def safe_roc_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)

scorer = make_scorer(safe_roc_auc, needs_proba=True)

    log.info(f"Starting sweep: {n_iter} trials x {cv_folds}-fold CV...")
    search = RandomizedSearchCV(
        base_xgb, PARAM_GRID,
        n_iter=n_iter, scoring=scorer, cv=cv,
        n_jobs=-1, verbose=1,
        random_state=random_state,
        return_train_score=True,
    )
    search.fit(X, y)

    # ── Save sweep log ────────────────────────────────────────────────────────
    results = pd.DataFrame(search.cv_results_)
    param_cols = [c for c in results.columns if c.startswith("param_")]
    keep = param_cols + ["mean_test_score", "std_test_score",
                         "mean_train_score", "rank_test_score", "mean_fit_time"]
    results = results[keep].copy()
    results.columns = [c.replace("param_", "") for c in results.columns]
    results = results.sort_values("rank_test_score").reset_index(drop=True)
    for col in ["mean_test_score", "std_test_score", "mean_train_score", "mean_fit_time"]:
        results[col] = results[col].round(4)
    results["swept_at"]         = datetime.now().strftime("%Y-%m-%d %H:%M")
    results["cv_folds"]         = cv_folds
    results["scale_pos_weight"] = spw

    results.to_csv(SWEEP_LOG, index=False)
    log.info(f"Sweep log saved → {SWEEP_LOG}")

    # ── Print top 5 ───────────────────────────────────────────────────────────
    best       = search.best_params_
    best_score = round(search.best_score_, 4)

    print("\n── Top 5 sweep trials ──────────────────────────────────────────────")
    print(results[["rank_test_score", "mean_test_score", "std_test_score",
                   "n_estimators", "max_depth", "learning_rate"]].head(5).to_string(index=False))
    print(f"\n── Best params  (CV ROC-AUC: {best_score}) ─────────────────────────")
    for k, v in best.items():
        print(f"  {k:<22} {v}")

    # ── Retrain full ensemble with best XGB params ────────────────────────────
    log.info("Retraining full ensemble with best params...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_xgb = XGBClassifier(
        **best,
        eval_metric="logloss",
        scale_pos_weight=spw,
        random_state=random_state,
        verbosity=0,
    )
    rf  = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=2,
        class_weight="balanced", random_state=random_state, n_jobs=-1,
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 128), activation="relu",
        max_iter=300, early_stopping=True, random_state=random_state,
    )
    ensemble = VotingClassifier(
        estimators=[("xgb", best_xgb), ("rf", rf), ("mlp", mlp)],
        voting="soft", n_jobs=-1,
    )

    model         = DrugRepurposingModel(random_state=random_state)
    model.ensemble = ensemble
    model.scaler   = StandardScaler()
    X_scaled       = model.scaler.fit_transform(X_train)
    ensemble.fit(X_scaled, y_train)
    model._is_fitted    = True
    model.feature_names = engineer.feature_names

    metrics = model.evaluate(X_test, y_test)
    print("\n── Retrained ensemble metrics ──────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")

    model.save("drug_repurposing_v2_swept")
    print(f"\nBest model → models/drug_repurposing_v2_swept.joblib")
    print(f"Sweep log  → data/sweep_results.csv  ({len(results)} trials)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_iter", type=int, default=20)
    p.add_argument("--cv",     type=int, default=5)
    args = p.parse_args()
    run_sweep(n_iter=args.n_iter, cv_folds=args.cv)
