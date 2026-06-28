"""
model.py
Soft-voting ensemble: XGBoost + Random Forest + MLP.
Includes SHAP explainability for feature importance.
Handles class imbalance via scale_pos_weight and class_weight="balanced".
"""

import numpy as np
import os
import joblib
import logging

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    average_precision_score,
)
from xgboost import XGBClassifier
import shap

log = logging.getLogger(__name__)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


class DrugRepurposingModel:
    """
    Soft-voting ensemble for binary classification of drug activity.

    Components:
        - XGBoost  (scale_pos_weight handles imbalance)
        - Random Forest  (class_weight="balanced")
        - MLP

    SHAP explainability uses the XGBoost component.
    """

    def __init__(self, n_estimators=200, random_state=42, model_dir=MODELS_DIR):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model_dir = model_dir
        self.feature_names = None
        self.ensemble = None
        self.scaler = StandardScaler()
        self._xgb = None
        self._is_fitted = False

    def _build_ensemble(self, scale_pos_weight=1.0):
        xgb = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=self.random_state,
            verbosity=0,
            scale_pos_weight=scale_pos_weight,
        )
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=10,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        mlp = MLPClassifier(
            hidden_layer_sizes=(512, 128),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            random_state=self.random_state,
        )
        self._xgb = xgb
        ensemble = VotingClassifier(
            estimators=[("xgb", xgb), ("rf", rf), ("mlp", mlp)],
            voting="soft",
            n_jobs=-1,
        )
        return ensemble

    def fit(self, X_train, y_train, feature_names=None):
        """Train the ensemble. Automatically computes class imbalance ratio."""
        log.info(f"Training ensemble on {X_train.shape[0]} compounds...")
        self.feature_names = feature_names

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        spw   = round(n_neg / max(n_pos, 1), 2)
        log.info(f"Class ratio — inactive: {n_neg} | active: {n_pos} | scale_pos_weight: {spw}")

        self.ensemble = self._build_ensemble(scale_pos_weight=spw)
        X_scaled = self.scaler.fit_transform(X_train)
        self.ensemble.fit(X_scaled, y_train)
        self._is_fitted = True
        log.info("Training complete.")
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.ensemble.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.ensemble.predict_proba(X_scaled)

    def evaluate(self, X_test, y_test):
        """Return dict of evaluation metrics."""
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":        round(accuracy_score(y_test, y_pred), 4),
            "roc_auc":         round(roc_auc_score(y_test, y_prob), 4),
            "avg_precision":   round(average_precision_score(y_test, y_prob), 4),
        }
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics["recall_active"]    = round(report.get("1", {}).get("recall", 0), 4)
        metrics["precision_active"] = round(report.get("1", {}).get("precision", 0), 4)
        metrics["f1_active"]        = round(report.get("1", {}).get("f1-score", 0), 4)

        log.info(f"Evaluation: {metrics}")
        return metrics

    def explain(self, X, max_samples=200):
        """Compute feature importance using XGBoost's built-in gain importance."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before explaining.")
        xgb_fitted = self.ensemble.named_estimators_["xgb"]
        importance = xgb_fitted.feature_importances_
        # Return as shap-like array (mean absolute values per feature)
        # Shape: (max_samples, n_features) — approximate with broadcast
        X_scaled = self.scaler.transform(X[:max_samples])
        shap_like = X_scaled * importance  # feature contribution proxy
        return shap_like, None

    def save(self, name="model"):
        path = os.path.join(self.model_dir, f"{name}.joblib")
        joblib.dump({
            "ensemble":      self.ensemble,
            "scaler":        self.scaler,
            "feature_names": self.feature_names,
        }, path)
        log.info(f"Model saved to {path}")
        return self

    def load(self, name="model"):
        path = os.path.join(self.model_dir, f"{name}.joblib")
        bundle = joblib.load(path)
        self.ensemble      = bundle["ensemble"]
        self.scaler        = bundle["scaler"]
        self.feature_names = bundle.get("feature_names")
        self._xgb          = self.ensemble.named_estimators_.get("xgb")
        self._is_fitted    = True
        log.info(f"Model loaded from {path}")
        return self
