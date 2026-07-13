# 🧬 COVID-19 Drug Repurposing ML Pipeline

End-to-end machine learning pipeline for identifying COVID-19 therapeutic candidates from FDA-approved compounds using cheminformatics and ensemble ML.

**ROC-AUC 0.855 · F1 0.712 · 1,620 compounds · 5-page Streamlit dashboard**

---

## Results

| Metric | v1 (default) | v2 (swept) |
|---|---|---|
| ROC-AUC | 0.8525 | **0.8553** |
| F1 (active class) | 0.6667 | **0.7125** |
| Precision | 0.7067 | **0.7500** |
| Recall | 0.6310 | **0.6786** |

Known COVID drugs correctly identified: ✓ Remdesivir · ✓ Dexamethasone · ✓ Nirmatrelvir · ✓ Molnupiravir

---

## Stack

| Layer | Tools |
|---|---|
| Data | ChEMBL REST API · PubChem PUG REST (4 COVID bioassays) |
| Features | RDKit ECFP4 Morgan fingerprints (2048 bits) + 15 molecular descriptors = 2063 features |
| Model | Soft-voting ensemble: XGBoost + Random Forest + MLP |
| Explainability | XGBoost gain-based feature importance |
| Hyperparameter tuning | RandomizedSearchCV · 20 trials · 5-fold stratified CV |
| Dashboard | Streamlit (5 pages) |

---

## Setup

```bash
conda create -n wave_prep python=3.10 -y
conda activate wave_prep
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

## Run

```bash
python train.py       # fetch data + train baseline model (~5 min first run)
python sweep.py       # hyperparameter sweep, 20 trials (~5 min)
streamlit run app.py  # launch dashboard at localhost:8501
```

---

## Project structure

```
covid_drug_repurposing/
├── data_pipeline.py       # ChEMBL + PubChem API fetching and ETL
├── feature_engineering.py # RDKit ECFP4 fingerprints + 15 descriptors
├── model.py               # XGBoost/RF/MLP soft-voting ensemble
├── train.py               # End-to-end training script
├── sweep.py               # RandomizedSearchCV hyperparameter sweep
├── app.py                 # Streamlit dashboard (5 pages)
├── requirements.txt
├── data/
│   ├── compounds.csv           # fetched dataset (gitignored)
│   └── sweep_results.csv       # hyperparameter sweep log
└── models/
    ├── drug_repurposing_v1.joblib        # baseline model (gitignored)
    └── drug_repurposing_v2_swept.joblib  # best model after sweep (gitignored)
```

---

## Dashboard pages

| Page | Description |
|---|---|
| Overview | Dataset stats, label distribution, key metrics |
| Predict a compound | Enter any SMILES, get activity probability + gauge dial |
| Feature importance | Top 20 XGBoost features driving predictions |
| Top candidates | All compounds ranked by predicted activity |
| Sweep results | Hyperparameter trials, before/after comparison table |

---

## Key engineering decisions

**Class imbalance** — FDA-approved compounds are overwhelmingly inactive against COVID. Fixed via `scale_pos_weight = n_inactive / n_active` in XGBoost and `class_weight="balanced"` in Random Forest. Without this, the model achieved 99.7% accuracy by predicting everything inactive.

**PubChem API** — SMILES fetching requires POST requests returning `ConnectivitySMILES` (not GET with `CanonicalSMILES`). Discovered by debugging zero-record returns despite 13,000+ active CIDs retrieved.

**Top predictive feature** — `NumAromaticRings` is the strongest driver, consistent with antiviral scaffold chemistry where aromatic cores dominate active compounds.

---

## Relevance to oligonucleotide drug discovery

This pipeline demonstrates the same ML workflow used in oligonucleotide QSAR modeling: feature engineering from molecular structure, ensemble classification with class imbalance handling, and explainability for scientific partners. The natural extension for RNA therapeutics is sequence-aware featurization — k-mer encodings, backbone modification flags (PS/PN/PO), and stereochemistry pattern encoding — alongside physicochemical descriptors.
