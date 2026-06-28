"""
feature_engineering.py
Converts SMILES strings into ML-ready features using RDKit.
Generates ECFP4 Morgan fingerprints + 15 molecular descriptors.
"""

import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # suppress RDKit warnings
log = logging.getLogger(__name__)


DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors",
    "TPSA", "NumRotatableBonds", "NumAromaticRings",
    "NumHeavyAtoms", "RingCount", "FractionCSP3",
    "NumAliphaticRings", "NumSaturatedRings",
    "MaxPartialCharge", "MinPartialCharge", "qed",
]


def smiles_to_mol(smiles):
    """Safely parse SMILES to RDKit mol object."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def mol_to_ecfp4(mol, n_bits=2048):
    """Generate ECFP4 Morgan fingerprint as numpy array."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp)


def mol_to_descriptors(mol):
    """Compute 15 molecular descriptors. Returns None if computation fails."""
    try:
        from rdkit.Chem import QED
        desc = {
            "MolWt":              Descriptors.MolWt(mol),
            "MolLogP":            Descriptors.MolLogP(mol),
            "NumHDonors":         rdMolDescriptors.CalcNumHBD(mol),
            "NumHAcceptors":      rdMolDescriptors.CalcNumHBA(mol),
            "TPSA":               rdMolDescriptors.CalcTPSA(mol),
            "NumRotatableBonds":  rdMolDescriptors.CalcNumRotatableBonds(mol),
            "NumAromaticRings":   rdMolDescriptors.CalcNumAromaticRings(mol),
            "NumHeavyAtoms":      mol.GetNumHeavyAtoms(),
            "RingCount":          rdMolDescriptors.CalcNumRings(mol),
            "FractionCSP3":       rdMolDescriptors.CalcFractionCSP3(mol),
            "NumAliphaticRings":  rdMolDescriptors.CalcNumAliphaticRings(mol),
            "NumSaturatedRings":  rdMolDescriptors.CalcNumSaturatedRings(mol),
            "MaxPartialCharge":   Descriptors.MaxPartialCharge(mol),
            "MinPartialCharge":   Descriptors.MinPartialCharge(mol),
            "qed":                QED.qed(mol),
        }
        return desc
    except Exception:
        return None


class MolecularFeatureEngineer:
    """
    Transforms a DataFrame with a 'smiles' column into
    (X, y) arrays ready for model training.

    X = ECFP4 fingerprint (2048 bits) + 15 descriptors = 2063 features
    y = binary label (1 = active, 0 = inactive)
    """

    def __init__(self, ecfp_bits=2048):
        self.ecfp_bits = ecfp_bits
        self.feature_names = None

    def fit_transform(self, df):
        """
        Args:
            df: DataFrame with 'smiles' and 'active' columns
        Returns:
            X: np.ndarray (n_valid_compounds, n_features)
            y: np.ndarray (n_valid_compounds,)
        """
        X_rows, y_rows, names = [], [], []
        failed = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing"):
            mol = smiles_to_mol(str(row["smiles"]))
            if mol is None:
                failed += 1
                continue

            fp = mol_to_ecfp4(mol, n_bits=self.ecfp_bits)
            desc = mol_to_descriptors(mol)
            if desc is None:
                failed += 1
                continue

            desc_vals = np.array([desc[k] for k in DESCRIPTOR_NAMES], dtype=np.float32)

            # Replace any NaN/inf in descriptors with 0
            desc_vals = np.nan_to_num(desc_vals, nan=0.0, posinf=0.0, neginf=0.0)

            X_rows.append(np.concatenate([fp, desc_vals]))
            y_rows.append(int(row["active"]))
            names.append(row.get("name", "unknown"))

        log.info(f"Featurized {len(X_rows)} compounds. Failed/skipped: {failed}")

        # Store feature names for later use (SHAP, etc.)
        fp_names = [f"fp_{i}" for i in range(self.ecfp_bits)]
        self.feature_names = fp_names + DESCRIPTOR_NAMES

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.int32)
        return X, y

    def transform_single(self, smiles):
        """Featurize a single SMILES string. Returns feature vector or None."""
        mol = smiles_to_mol(smiles)
        if mol is None:
            return None
        fp = mol_to_ecfp4(mol, n_bits=self.ecfp_bits)
        desc = mol_to_descriptors(mol)
        if desc is None:
            return None
        desc_vals = np.array([desc[k] for k in DESCRIPTOR_NAMES], dtype=np.float32)
        desc_vals = np.nan_to_num(desc_vals, nan=0.0, posinf=0.0, neginf=0.0)
        return np.concatenate([fp, desc_vals]).reshape(1, -1)


if __name__ == "__main__":
    import os
    data_path = os.path.join(os.path.dirname(__file__), "data", "compounds.csv")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} compounds")

    eng = MolecularFeatureEngineer()
    X, y = eng.fit_transform(df)
    print(f"Feature matrix: {X.shape}")
    print(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")
