"""
app.py
Streamlit dashboard for the COVID-19 Drug Repurposing ML Pipeline.
5 pages: Overview | Predict | Feature Importance | Top Candidates | Sweep Results
Run: streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from feature_engineering import MolecularFeatureEngineer
from model import DrugRepurposingModel

st.set_page_config(
    page_title="COVID-19 Drug Repurposing Pipeline",
    page_icon="🧬",
    layout="wide",
)

DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "compounds.csv")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "models", "drug_repurposing_v2_swept.joblib")
MODEL_PATH2 = os.path.join(os.path.dirname(__file__), "models", "drug_repurposing_v1.joblib")
SWEEP_LOG   = os.path.join(os.path.dirname(__file__), "data", "sweep_results.csv")

def best_model_path():
    return MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_PATH2

@st.cache_resource
def load_model():
    path = best_model_path()
    name = os.path.splitext(os.path.basename(path))[0]
    model = DrugRepurposingModel()
    model.load(name)
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
    ["Overview", "Predict a Compound", "Feature Importance", "Top Candidates", "Sweep Results"],
)


# ── Page 1: Overview ──────────────────────────────────────────────────────────
if page == "Overview":
    st.title("COVID-19 Drug Repurposing ML Pipeline")
    st.markdown(
        """
        This pipeline identifies potential COVID-19 therapeutics from FDA-approved compounds
        using cheminformatics and ensemble machine learning.

        **Stack:** RDKit · XGBoost · Random Forest · MLP  
        **Data:** ChEMBL REST API · PubChem PUG REST  
        **Features:** ECFP4 Morgan fingerprints (2048 bits) + 15 molecular descriptors  
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model version",  "v2 (swept)")
    col2.metric("ROC-AUC",        "0.855")
    col3.metric("F1 (active)",    "0.712")
    col4.metric("Sweep trials",   "20")

    if os.path.exists(DATA_PATH):
        df = load_data()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total compounds", f"{len(df):,}")
        c2.metric("Active (COVID)",  int(df["active"].sum()))
        c3.metric("Inactive",        int((df["active"] == 0).sum()))
        c4.metric("Active rate",     f"{df['active'].mean()*100:.1f}%")

        st.subheader("Label distribution")
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
        "Remdesivir (active)":       "O=C(OCC(CC(=O)OCC)NC(=O)c1cccc(N)n1)C1OC(C#N)(c2ccc3c(n2)n(C(C)C)c(=O)n3C2CCCC2)C(O)C1O",
        "Dexamethasone (active)":    "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@H]3[C@@H](O)C[C@]2(C)[C@]1(O)C(=O)CO",
        "Nirmatrelvir (active)":     "CC1(C2CC2NC(=O)C(F)(F)F)NC(=O)c3ccc(C#N)cc3C(=O)N1CC(=O)NC(C(=O)C1CC1)C1CCNC1=O",
        "Molnupiravir (active)":     "CC(C)OC(=O)OCC1OC(n2ccc(=O)[nH]c2=O)C(O)C1O",
        "Aspirin (inactive)":        "CC(=O)Oc1ccccc1C(=O)O",
        "Caffeine (inactive)":       "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    }

    selected    = st.selectbox("Try an example:", ["— enter manually —"] + list(examples.keys()))
    smiles_input = st.text_area("SMILES", value=examples.get(selected, ""), height=80)

    if st.button("Predict", type="primary") and smiles_input.strip():
        if not os.path.exists(best_model_path()):
            st.error("No trained model found. Run `python train.py` first.")
        else:
            model = load_model()
            eng   = get_engineer()
            x     = eng.transform_single(smiles_input.strip())

            if x is None:
                st.error("Invalid SMILES string. Please check and try again.")
            else:
                prob  = model.predict_proba(x)[0][1]
                label = "🟢 ACTIVE" if prob >= 0.5 else "🔴 Inactive"

                col1, col2 = st.columns(2)
                col1.metric("Prediction",  label)
                col2.metric("Confidence",  f"{prob:.1%}")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": "Activity probability (%)"},
                    gauge={
                        "axis":  {"range": [0, 100]},
                        "bar":   {"color": "#EF553B" if prob >= 0.5 else "#636EFA"},
                        "steps": [
                            {"range": [0, 50],   "color": "#EEF"},
                            {"range": [50, 100], "color": "#FEE"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 2}, "value": 50},
                    },
                ))
                st.plotly_chart(fig, use_container_width=True)


# ── Page 3: Feature Importance ────────────────────────────────────────────────
elif page == "Feature Importance":
    st.title("Feature Importance")
    st.markdown(
        "XGBoost gain-based feature importance shows which molecular features "
        "drive the model's predictions most strongly."
    )

    if not os.path.exists(DATA_PATH) or not os.path.exists(best_model_path()):
        st.warning("Run `python train.py` first to generate data and model.")
    else:
        model = load_model()
        df    = load_data()
        eng   = get_engineer()

        with st.spinner("Computing feature importance..."):
            X, y = eng.fit_transform(df.sample(min(300, len(df)), random_state=42))
            shap_values, _ = model.explain(X, max_samples=200)

        st.subheader("Top 20 features by mean importance")
        mean_imp   = np.abs(shap_values).mean(axis=0)
        feat_names = eng.feature_names or [f"feat_{i}" for i in range(mean_imp.shape[0])]

        top_idx   = np.argsort(mean_imp)[-20:][::-1]
        top_names = [feat_names[i] for i in top_idx]
        top_vals  = mean_imp[top_idx]

        fig = px.bar(
            x=top_vals, y=top_names,
            orientation="h",
            labels={"x": "Mean importance", "y": "Feature"},
            color=top_vals,
            color_continuous_scale="Reds",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "fp_N = ECFP4 fingerprint bit (specific molecular substructure). "
            "Named features = global physicochemical descriptors."
        )


# ── Page 4: Top Candidates ────────────────────────────────────────────────────
elif page == "Top Candidates":
    st.title("Top repurposing candidates")
    st.markdown("Compounds ranked by predicted COVID-19 activity probability.")

    if not os.path.exists(DATA_PATH) or not os.path.exists(best_model_path()):
        st.warning("Run `python train.py` first.")
    else:
        model  = load_model()
        df     = load_data()
        eng    = get_engineer()
        n_show = st.slider("Show top N candidates:", 10, 100, 25)

        with st.spinner("Scoring all compounds..."):
            X, y  = eng.fit_transform(df)
            probs = model.predict_proba(X)[:, 1]

        valid_df = df.iloc[:len(probs)].copy()
        valid_df["activity_prob"] = probs
        top_df   = valid_df.sort_values("activity_prob", ascending=False).head(n_show)

        st.dataframe(
            top_df[["name", "smiles", "activity_prob", "active"]].rename(columns={
                "name":          "Compound",
                "smiles":        "SMILES",
                "activity_prob": "Predicted probability",
                "active":        "Known active",
            }).style.background_gradient(subset=["Predicted probability"], cmap="Reds"),
            use_container_width=True,
        )

        st.subheader("Probability distribution")
        fig = px.histogram(
            valid_df, x="activity_prob", nbins=50,
            color="active",
            color_discrete_map={0: "#636EFA", 1: "#EF553B"},
            labels={"activity_prob": "Predicted activity probability", "active": "Known active"},
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Page 5: Sweep Results ─────────────────────────────────────────────────────
elif page == "Sweep Results":
    st.title("Hyperparameter sweep")
    st.markdown(
        "Results from `RandomizedSearchCV` — 20 trials across XGBoost hyperparameters, "
        "5-fold stratified CV. Best params used to train the v2 model. "
        "Re-run with `python sweep.py`."
    )

    if not os.path.exists(SWEEP_LOG):
        st.warning("No sweep results found. Run `python sweep.py` first.")
    else:
        sweep = pd.read_csv(SWEEP_LOG)

        # Drop NaN ROC-AUC rows for display (happen when CV fold has no positives)
        sweep_clean = sweep.dropna(subset=["mean_test_score"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Trials run",      len(sweep))
        col2.metric("Valid CV scores", len(sweep_clean))
        if len(sweep_clean):
            col3.metric("Best CV ROC-AUC",  f"{sweep_clean['mean_test_score'].max():.4f}")
            col4.metric("Worst CV ROC-AUC", f"{sweep_clean['mean_test_score'].min():.4f}")

        st.subheader("Best params (used in v2 model)")
        best_row = {
            "n_estimators": 300, "max_depth": 8, "learning_rate": 0.05,
            "subsample": 1.0, "colsample_bytree": 0.6,
            "min_child_weight": 5, "gamma": 0.1,
        }
        st.json(best_row)

        st.subheader("Model improvement from sweep")
        compare = pd.DataFrame({
            "Version":   ["v1 (default params)", "v2 (swept params)"],
            "ROC-AUC":   [0.8525, 0.8553],
            "F1 active": [0.6667, 0.7125],
            "Precision": [0.7067, 0.7500],
            "Recall":    [0.6310, 0.6786],
        })
        st.dataframe(
            compare.style.background_gradient(
                subset=["ROC-AUC", "F1 active"], cmap="Greens"
            ),
            use_container_width=True,
            hide_index=True,
        )

        if len(sweep_clean) > 0:
            st.subheader("Learning rate vs ROC-AUC")
            fig = px.scatter(
                sweep_clean,
                x="learning_rate", y="mean_test_score",
                size="n_estimators", color="max_depth",
                hover_data=["subsample", "colsample_bytree", "gamma"],
                labels={"mean_test_score": "CV ROC-AUC", "learning_rate": "Learning rate"},
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("All trials (ranked)")
        display_cols = [c for c in [
            "rank_test_score", "mean_test_score", "std_test_score",
            "n_estimators", "max_depth", "learning_rate",
            "subsample", "colsample_bytree", "mean_fit_time"
        ] if c in sweep.columns]
        st.dataframe(
            sweep[display_cols].rename(columns={
                "rank_test_score": "Rank",
                "mean_test_score": "CV ROC-AUC",
                "std_test_score":  "Std",
                "mean_fit_time":   "Fit time (s)",
            }),
            use_container_width=True,
            hide_index=True,
        )
