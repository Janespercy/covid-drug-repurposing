"""
data_pipeline.py
Fetches FDA-approved compounds and COVID-19 bioactivity data from
ChEMBL and PubChem public APIs. No API key required.

Fix: fetch actives directly from COVID bioassay endpoints rather than
cross-joining ChEMBL compound list with target bioactivity IDs.
"""

import requests
import pandas as pd
import numpy as np
import time
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHEMBL_BASE  = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ── Known COVID actives (curated) ─────────────────────────────────────────────
KNOWN_ACTIVES = {
    "remdesivir":      "O=C(OCC(CC(=O)OCC)NC(=O)c1cccc(N)n1)C1OC(C#N)(c2ccc3c(n2)n(C(C)C)c(=O)n3C2CCCC2)C(O)C1O",
    "dexamethasone":   "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@H]3[C@@H](O)C[C@]2(C)[C@]1(O)C(=O)CO",
    "nirmatrelvir":    "CC1(C2CC2NC(=O)C(F)(F)F)NC(=O)c3ccc(C#N)cc3C(=O)N1CC(=O)NC(C(=O)C1CC1)C1CCNC1=O",
    "molnupiravir":    "CC(C)OC(=O)OCC1OC(n2ccc(=O)[nH]c2=O)C(O)C1O",
    "baricitinib":     "CCS(=O)(=O)Nc1cccc(-c2nc(NC3CCNCC3)c3cncn3n2)c1",
    "casirivimab_ref": None,  # mAb, skip
    "lopinavir":       "CC(C)(C)NC(=O)[C@@H]1CN(Cc2ccccc2)CCN1CC(O)c1ccc(N)cc1",
    "ritonavir":       "CC(C)(C)NC(=O)c1nc(N(C)C(=O)[C@@H](Cc2ccccc2)NC(=O)c2nc(C(C)C)cs2)cs1",
    "favipiravir":     "NC(=O)c1ncc(F)c(=O)[nH]1",
    "ivermectin":      "C[C@@H]1O[C@@](O)(C2CC(=O)[C@H](C/C=C/[C@@H]3O[C@]4(O)C[C@@H](C[C@H]4[C@@H]3C)O[C@@H]3C[C@H](OC)[C@@H](O)[C@H](C)O3)[C@@H]2C)C[C@H]1O[C@@H]1C[C@H](OC)[C@@H](O)[C@H](C)O1",
    "chloroquine":     "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
    "hydroxychloroquine": "CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
    "azithromycin":    "CCC1OC(=O)[C@H](C)[C@@H](O[C@@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@@H](N(C)C)[C@H]2O)[C@@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O",
    "tocilizumab_ref": None,
    "colchicine":      "COc1ccc2c(c1OC)-c1cc(OC)c(=O)cc1[C@@H](NC(C)=O)C2",
    "dexlansoprazole": "CC1=CN=C(CS(=O)c2nc3cc(OCC(F)F)ccn23)C(OCC(F)F)=C1",
    "nafamostat":      "NC(=N)c1ccc(OC(=O)c2ccc3cc(C(=N)N)ccc3c2)cc1",
    "camostat":        "CN(C)C(=O)COC(=O)c1ccc(OC(=O)c2cccc(C(=N)N)c2)cc1",
    "aprotinin":       None,  # peptide, skip
    "ribavirin":       "NC(=O)c1ncn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)n1",
    "interferon_ref":  None,
    "tocilizumab2":    None,
    "emetine":         "COc1ccc(C[C@@H]2CN3CCc4cc(OC)c(OC)cc4[C@@H]3C[C@@H]2NCC)cc1OC",
    "cyclosporine":    "CCC1NC(=O)C(C(O)C(C)C/C=C/C)N(C)C(=O)C(C(C)CC)N(C)C(=O)C(CC(C)C)NC(=O)C(CC(C)C)N(C)C(=O)CN(C)C(=O)C(C)NC(=O)C(C)NC(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)NC(=O)C(CO)N(C)C1=O",
    "apilimod":        "Cc1cc(-c2ccnc(NCCOC)n2)cc(C)c1Nc1nc(N2CCOCC2)nc2ccccc12",
    "saracatinib":     "COc1cc2ncnc(Nc3ccc(F)cc3Cl)c2cc1OCCCN1CCOCC1",
}


# ── ChEMBL: fetch FDA-approved inactives ──────────────────────────────────────

def fetch_chembl_inactives(max_compounds=1200):
    """Fetch FDA-approved small molecules to serve as background inactives."""
    log.info("Fetching FDA-approved compounds from ChEMBL (inactives background)...")
    compounds, offset, limit = [], 0, 100

    with tqdm(total=max_compounds, desc="ChEMBL inactives") as pbar:
        while len(compounds) < max_compounds:
            url = (
                f"{CHEMBL_BASE}/molecule.json"
                f"?max_phase=4&molecule_type=Small+molecule"
                f"&limit={limit}&offset={offset}"
            )
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                molecules = r.json().get("molecules", [])
                if not molecules:
                    break
                for mol in molecules:
                    smi = (mol.get("molecule_structures") or {}).get("canonical_smiles")
                    if smi:
                        compounds.append({
                            "name":   mol.get("pref_name", "unknown"),
                            "smiles": smi,
                            "active": 0,
                        })
                pbar.update(len(molecules))
                offset += limit
                time.sleep(0.2)
            except Exception as e:
                log.warning(f"ChEMBL fetch error at offset {offset}: {e}")
                break

    return pd.DataFrame(compounds[:max_compounds])


# ── PubChem: fetch COVID bioassay actives ─────────────────────────────────────

def fetch_pubchem_covid_actives(max_cids=400):
    """
    Fetch active CIDs from NCATS COVID-19 bioassays on PubChem,
    then retrieve their SMILES.
    """
    log.info("Fetching COVID-19 actives from PubChem bioassays...")
    # Validated NCATS SARS-CoV-2 assays
    assay_aids = [
        "1851",   # SARS-CoV-2 cytopathic effect (Vero E6)
        "1706",   # SARS-CoV-2 CPE counter screen
        "1645871",# 3CLpro inhibition
        "1645840",# RdRp inhibition
    ]
    all_cids = []
    for aid in assay_aids:
        url = f"{PUBCHEM_BASE}/assay/aid/{aid}/cids/JSON?cids_type=active"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            cids = (r.json()
                    .get("InformationList", {})
                    .get("Information", [{}])[0]
                    .get("CID", []))
            all_cids.extend(cids)
            log.info(f"  AID {aid}: {len(cids)} active CIDs")
            time.sleep(0.3)
        except Exception as e:
            log.warning(f"PubChem AID {aid} error: {e}")

    all_cids = list(set(all_cids))[:max_cids]
    if not all_cids:
        log.warning("No PubChem CIDs retrieved.")
        return pd.DataFrame()

    # Use POST to avoid URL length limits; smaller batches to stay under rate limits
    records, batch_size = [], 50
    post_url = f"{PUBCHEM_BASE}/compound/cid/property/ConnectivitySMILES,MolecularWeight/JSON"
    for i in tqdm(range(0, len(all_cids), batch_size), desc="PubChem SMILES"):
        batch   = all_cids[i:i + batch_size]
        cid_str = ",".join(map(str, batch))
        try:
            r = requests.post(post_url, data={"cid": cid_str}, timeout=30)
            r.raise_for_status()
            props = r.json().get("PropertyTable", {}).get("Properties", [])
            for p in props:
                records.append({
                    "name":   f"pubchem_{p['CID']}",
                    "smiles": p.get("ConnectivitySMILES"),
                    "active": 1,
                })
            time.sleep(0.3)
        except Exception as e:
            log.warning(f"PubChem SMILES batch error (CIDs {i}-{i+batch_size}): {e}")

    if not records:
        log.warning("PubChem SMILES fetch returned 0 records — check API connectivity.")
        return pd.DataFrame(columns=["name", "smiles", "active"])
    df = pd.DataFrame(records).dropna(subset=["smiles"])
    log.info(f"PubChem actives with SMILES: {len(df)}")
    return df


# ── ChEMBL: fetch COVID bioactivity actives directly ─────────────────────────

def fetch_chembl_covid_actives(max_records=300):
    """
    Fetch compounds directly from ChEMBL antiviral bioactivity data
    for SARS-CoV-2. Uses activity endpoint filtered by target organism.
    """
    log.info("Fetching COVID-19 actives from ChEMBL bioactivity...")
    records = []
    url = (
        f"{CHEMBL_BASE}/activity.json"
        f"?target_organism=SARS-CoV-2"
        f"&standard_type=IC50"
        f"&standard_value__lte=1000"
        f"&limit=200&offset=0"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        activities = r.json().get("activities", [])
        log.info(f"  ChEMBL SARS-CoV-2 IC50 ≤1µM: {len(activities)} records")
        for act in activities:
            smi = act.get("canonical_smiles")
            if smi:
                records.append({
                    "name":   act.get("molecule_chembl_id", "unknown"),
                    "smiles": smi,
                    "active": 1,
                })
        time.sleep(0.3)
    except Exception as e:
        log.warning(f"ChEMBL COVID actives fetch error: {e}")

    if not records:
        log.warning("No ChEMBL COVID actives returned.")
        return pd.DataFrame(columns=["name", "smiles", "active"])
    df = pd.DataFrame(records).dropna(subset=["smiles"])
    log.info(f"ChEMBL COVID actives: {len(df)}")
    return df


# ── Build final dataset ───────────────────────────────────────────────────────

def build_dataset():
    """
    Combine:
      1. ChEMBL FDA-approved compounds  → inactives (background)
      2. PubChem COVID bioassay actives → actives
      3. ChEMBL SARS-CoV-2 IC50 actives → actives
      4. Curated known actives          → actives

    Saves to data/compounds.csv and returns DataFrame.
    """
    # Inactives background
    inactive_df = fetch_chembl_inactives(max_compounds=1200)

    # Actives from multiple sources
    pubchem_df  = fetch_pubchem_covid_actives(max_cids=400)
    chembl_act  = fetch_chembl_covid_actives(max_records=300)

    # Curated known actives
    curated = [
        {"name": name, "smiles": smi, "active": 1}
        for name, smi in KNOWN_ACTIVES.items()
        if smi is not None
    ]
    curated_df = pd.DataFrame(curated)

    frames = [inactive_df]
    if not pubchem_df.empty:
        frames.append(pubchem_df)
    if not chembl_act.empty:
        frames.append(chembl_act)
    frames.append(curated_df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["smiles"])
    combined = combined.drop_duplicates(subset=["smiles"])
    combined = combined.reset_index(drop=True)

    n_active = int(combined["active"].sum())
    log.info(
        f"Final dataset: {len(combined)} compounds | "
        f"Actives: {n_active} | Inactives: {len(combined) - n_active}"
    )
    out_path = os.path.join(DATA_DIR, "compounds.csv")
    combined.to_csv(out_path, index=False)
    return combined


if __name__ == "__main__":
    df = build_dataset()
    print(df["active"].value_counts())
    print("\nActive compounds:")
    print(df[df["active"] == 1]["name"].tolist())
