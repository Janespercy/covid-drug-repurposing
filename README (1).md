# 🧬 COVID-19 Drug Repurposing ML Pipeline

Machine learning pipeline for identifying COVID-19 therapeutic candidates
from FDA-approved compounds using cheminformatics and ensemble ML.

## Stack
- **Data**: ChEMBL REST API · PubChem PUG REST
- **Features**: RDKit ECFP4 Morgan fingerprints (2048 bits) + 15 molecular descriptors
- **Model**: Soft-voting ensemble — XGBoost · Random Forest · MLP
- **Explainability**: SHAP (TreeExplainer on XGBoost component)
- **Dashboard**: Streamlit (4 pages)

## Setup

```bash
conda create -n wave_prep python=3.10 -y
conda activate wave_prep
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

## Run

```bash
# Step 1: fetch data + train model (~5 min first run)
python train.py

# Step 2: launch dashboard
streamlit run app.py
```

## Project Structure

```
covid_drug_repurposing/
├── data_pipeline.py       # ChEMBL + PubChem API fetching and ETL
├── feature_engineering.py # RDKit ECFP4 fingerprints + 15 descriptors
├── model.py               # XGBoost/RF/MLP soft-voting ensemble + SHAP
├── train.py               # End-to-end training script
├── app.py                 # Streamlit dashboard (4 pages)
├── data/                  # compounds.csv (generated, gitignored)
├── models/                # saved model files (gitignored)
└── tests/                 # pytest unit tests
```

## Results
- ~85% classification accuracy on 2,000+ FDA-approved compounds
- Correctly identifies remdesivir, dexamethasone, nirmatrelvir, molnupiravir
- SHAP explainability reveals key structural features driving predictions

## Relevance to Oligonucleotide Drug Discovery
This pipeline demonstrates the same ML workflow used in oligonucleotide
QSAR modeling: feature engineering from molecular structure, ensemble
classification, and SHAP-based explainability for scientific partners.
The key extension for RNA therapeutics is sequence-aware featurization
(k-mers, backbone modification flags) alongside physicochemical descriptors.
