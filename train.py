"""
train.py
End-to-end training script.
Run: python train.py
"""

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from data_pipeline import build_dataset
from feature_engineering import MolecularFeatureEngineer
from model import DrugRepurposingModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "compounds.csv")


def main():
    # ── Step 1: Data ──────────────────────────────────────────────────────────
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        n_active = int(df["active"].sum())
        log.info(f"Loaded cached dataset: {len(df)} compounds | Actives: {n_active}")

        # If actives are suspiciously low, re-fetch
        if n_active < 20:
            log.warning(f"Only {n_active} actives in cache — re-fetching fresh data...")
            os.remove(DATA_PATH)
            df = build_dataset()
    else:
        log.info("No cached data — fetching from APIs...")
        df = build_dataset()

    n_active   = int(df["active"].sum())
    n_inactive = int((df["active"] == 0).sum())
    log.info(f"Final dataset: {len(df)} compounds | Actives: {n_active} | Inactives: {n_inactive}")

    if n_active < 5:
        log.error("Fewer than 5 actives found — cannot train a meaningful model. Check API connectivity.")
        return

    # ── Step 2: Featurize ─────────────────────────────────────────────────────
    engineer = MolecularFeatureEngineer(ecfp_bits=2048)
    X, y = engineer.fit_transform(df)
    log.info(f"Feature matrix: {X.shape} | Actives in features: {y.sum()}")

    # ── Step 3: Train/test split ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    model = DrugRepurposingModel(n_estimators=200)
    model.fit(X_train, y_train, feature_names=engineer.feature_names)

    # ── Step 5: Evaluate ──────────────────────────────────────────────────────
    metrics = model.evaluate(X_test, y_test)
    print("\n── Evaluation Results ──────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")

    # Interpret ROC-AUC
    roc = metrics["roc_auc"]
    if roc > 0.75:
        verdict = "✓ Good discrimination"
    elif roc > 0.6:
        verdict = "~ Moderate — more actives would help"
    else:
        verdict = "✗ Poor — likely still imbalanced"
    print(f"\n  ROC-AUC verdict: {verdict}")

    # ── Step 6: Validate known COVID drugs ────────────────────────────────────
    print("\n── Known Drug Validation ───────────────────")
    known = {
        "Remdesivir":    "O=C(OCC(CC(=O)OCC)NC(=O)c1cccc(N)n1)C1OC(C#N)(c2ccc3c(n2)n(C(C)C)c(=O)n3C2CCCC2)C(O)C1O",
        "Dexamethasone": "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@H]3[C@@H](O)C[C@]2(C)[C@]1(O)C(=O)CO",
        "Nirmatrelvir":  "CC1(C2CC2NC(=O)C(F)(F)F)NC(=O)c3ccc(C#N)cc3C(=O)N1CC(=O)NC(C(=O)C1CC1)C1CCNC1=O",
        "Molnupiravir":  "CC(C)OC(=O)OCC1OC(n2ccc(=O)[nH]c2=O)C(O)C1O",
    }
    eng2 = MolecularFeatureEngineer(ecfp_bits=2048)
    for drug, smiles in known.items():
        x = eng2.transform_single(smiles)
        if x is not None:
            prob = model.predict_proba(x)[0][1]
            label = "✓ ACTIVE" if prob >= 0.5 else "✗ inactive"
            print(f"  {drug:<18} p={prob:.3f}  {label}")

    # ── Step 7: Save ──────────────────────────────────────────────────────────
    model.save("drug_repurposing_v1")
    print("\nModel saved to models/drug_repurposing_v1.joblib")
    print("Run: streamlit run app.py")


if __name__ == "__main__":
    main()
