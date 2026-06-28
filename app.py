"""
app.py
Streamlit dashboard for the COVID-19 Drug Repurposing ML Pipeline.
4 pages: Overview | Predict | SHAP Explainability | Top Candidates
Run: streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from feature_engineering import MolecularFeatureEngineer
from model import DrugRepurposingModel

st.set_page_config(
    page_title="COVID-19 Drug Repurposing Pipeline",
    page_icon="🧬",
    layout="wide",
)

DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "compounds.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "drug_repurposing_v1.joblib")


@st.cache_resource
def load_model():
    model = DrugRepurposingModel()
    model.load("drug_repurposing_v1")
    return model

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def get_engineer():
    return MolecularFeatureEngineer(ecfp_bits=2048)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🧬 Drug Repurposing")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Predict a Compound", "SHAP Explainability", "Top Candidates"],
)


# ── Page 1: Overview ─────────────────────────────────────────────────────────
if page == "Overview":
    st.title("COVID-19 Drug Repurposing ML Pipeline")
    st.markdown(
        """
        This pipeline identifies potential COVID-19 therapeutics from FDA-approved compounds
        using cheminformatics and ensemble machine learning.

        **Stack:** RDKit · XGBoost · Random Forest · MLP · SHAP  
        **Data:** ChEMBL REST API · PubChem PUG REST  
        **Features:** ECFP4 Morgan fingerprints (2048 bits) + 15 molecular descriptors  
        """
    )

    if os.path.exists(DATA_PATH):
        df = load_data()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Compounds", f"{len(df):,}")
        col2.metric("Active (COVID)", int(df["active"].sum()))
        col3.metric("Inactive", int((df["active"] == 0).sum()))
        col4.metric("Active Rate", f"{df['active'].mean()*100:.1f}%")

        st.subheader("Label Distribution")
        fig = px.pie(
            values=df["active"].value_counts().values,
            names=["Inactive", "Active"],
            color_discrete_sequence=["#636EFA", "#EF553B"],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No dataset found. Run `python train.py` first.")


# ── Page 2: Predict ───────────────────────────────────────────────────────────
elif page == "Predict a Compound":
    st.title("Predict COVID-19 Activity")
    st.markdown("Enter a SMILES string to predict whether the compound is active against SARS-CoV-2.")

    examples = {
        "Remdesivir (active)":    "O=C(OCC(CC(=O)OCC)NC(=O)c1cccc(N)n1)C1OC(C#N)(c2ccc3c(n2)n(C(C)C)c(=O)n3C2CCCC2)C(O)C1O",
        "Dexamethasone (active)": "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@H]3[C@@H](O)C[C@]2(C)[C@]1(O)C(=O)CO",
        "Aspirin (inactive)":     "CC(=O)Oc1ccccc1C(=O)O",
        "Caffeine (inactive)":    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    }

    selected = st.selectbox("Try an example:", ["— enter manually —"] + list(examples.keys()))
    smiles_input = st.text_area(
        "SMILES",
        value=examples.get(selected, ""),
        height=80,
    )

    if st.button("Predict", type="primary") and smiles_input.strip():
        if not os.path.exists(MODEL_PATH):
            st.error("No trained model found. Run `python train.py` first.")
        else:
            model = load_model()
            eng = get_engineer()
            x = eng.transform_single(smiles_input.strip())

            if x is None:
                st.error("Invalid SMILES string. Please check and try again.")
            else:
                prob = model.predict_proba(x)[0][1]
                label = "🟢 ACTIVE" if prob >= 0.5 else "🔴 Inactive"

                col1, col2 = st.columns(2)
                col1.metric("Prediction", label)
                col2.metric("Confidence", f"{prob:.1%}")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": "Activity Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "#EF553B" if prob >= 0.5 else "#636EFA"},
                        "steps": [
                            {"range": [0, 50],  "color": "#EEF"},
                            {"range": [50, 100], "color": "#FEE"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 2}, "value": 50},
                    },
                ))
                st.plotly_chart(fig, use_container_width=True)


# ── Page 3: SHAP ─────────────────────────────────────────────────────────────
elif page == "SHAP Explainability":
    st.title("SHAP Feature Importance")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) shows which molecular features "
        "drive the model's predictions. The XGBoost component is used for SHAP."
    )

    if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
        st.warning("Run `python train.py` first to generate data and model.")
    else:
        model = load_model()
        df = load_data()
        eng = get_engineer()

        with st.spinner("Computing feature importance..."):
            X, y = eng.fit_transform(df.sample(min(300, len(df)), random_state=42))
            shap_values, _ = model.explain(X, max_samples=200)

        st.subheader("Top 20 Features by Mean Importance")
        mean_shap = np.abs(shap_values).mean(axis=0)
        feat_names = eng.feature_names or [f"feat_{i}" for i in range(mean_shap.shape[0])]

        top_idx = np.argsort(mean_shap)[-20:][::-1]
        top_names = [feat_names[i] for i in top_idx]
        top_vals  = mean_shap[top_idx]

        fig = px.bar(
            x=top_vals, y=top_names,
            orientation="h",
            labels={"x": "Mean |SHAP value|", "y": "Feature"},
            color=top_vals,
            color_continuous_scale="Reds",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Fingerprint bits (fp_N) reflect specific molecular substructures. "
            "Descriptor features (MolWt, TPSA, etc.) reflect global physicochemical properties."
        )


# ── Page 4: Top Candidates ────────────────────────────────────────────────────
elif page == "Top Candidates":
    st.title("Top Repurposing Candidates")
    st.markdown("Compounds from the dataset ranked by predicted COVID-19 activity probability.")

    if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
        st.warning("Run `python train.py` first to generate data and model.")
    else:
        model = load_model()
        df = load_data()
        eng = get_engineer()

        n_show = st.slider("Show top N candidates:", 10, 100, 25)

        with st.spinner("Scoring all compounds..."):
            X, y = eng.fit_transform(df)
            probs = model.predict_proba(X)[:, 1]

        valid_df = df.iloc[:len(probs)].copy()
        valid_df["activity_prob"] = probs
        valid_df["prediction"] = (probs >= 0.5).astype(int)

        top_df = valid_df.sort_values("activity_prob", ascending=False).head(n_show)

        st.dataframe(
            top_df[["name", "smiles", "activity_prob", "active"]].rename(columns={
                "name": "Compound",
                "smiles": "SMILES",
                "activity_prob": "Predicted Probability",
                "active": "Known Active",
            }).style.background_gradient(subset=["Predicted Probability"], cmap="Reds"),
            use_container_width=True,
        )

        st.subheader("Probability Distribution")
        fig = px.histogram(
            valid_df, x="activity_prob", nbins=50,
            color="active",
            color_discrete_map={0: "#636EFA", 1: "#EF553B"},
            labels={"activity_prob": "Predicted Activity Probability", "active": "Known Active"},
        )
        st.plotly_chart(fig, use_container_width=True)
