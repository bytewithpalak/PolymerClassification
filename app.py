import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Polymer Classifier",
    page_icon="🧪",
    layout="wide",
)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🧪 Polymer Classification App")
st.markdown(
    "Upload a molecular fingerprint dataset (CSV with 2048 binary features) "
    "to predict whether each molecule is a **Plastic**, **Peptide**, or **Oligosaccharide**."
)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    pipeline = load_model()
except FileNotFoundError:
    st.error("❌ `model.pkl` not found. Place it in the same folder as this app.")
    st.stop()

# Label map – XGBoost encodes classes alphabetically: 0=oligosaccharide, 1=peptide, 2=plastic
LABEL_MAP = {0: "Oligosaccharide", 1: "Peptide", 2: "Plastic"}
CLASS_COLORS = {
    "Oligosaccharide": "#636EFA",
    "Peptide":         "#EF553B",
    "Plastic":         "#00CC96",
}

st.divider()

# ── Sidebar – info ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        "**Model:** XGBoost (trained on 20 609 molecules)\n\n"
        "**Features:** 2 048-bit Morgan fingerprints\n\n"
        "**Classes:**\n"
        "- 🟢 Plastic\n"
        "- 🔴 Peptide\n"
        "- 🔵 Oligosaccharide\n\n"
        "**Accuracy:** ~100 % on test set"
    )
    st.divider()
    st.markdown("**Expected CSV columns:**")
    st.code("smiles (optional)\n0, 1, 2, … 2047  (fingerprint bits)")

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Upload CSV file", type=["csv"],
    help="CSV must contain 2 048 binary fingerprint columns (named 0–2047)."
)

if uploaded_file is None:
    st.info("👆 Upload a CSV file to get started.")
    st.stop()

# ── Load & validate ───────────────────────────────────────────────────────────
df_raw = pd.read_csv(uploaded_file)

st.subheader("📄 Uploaded data preview")
st.dataframe(df_raw.head(10), use_container_width=True)

# Separate metadata from fingerprint features
meta_cols = [c for c in ["label", "smiles", "Unnamed: 0"] if c in df_raw.columns]
X = df_raw.drop(columns=meta_cols)

# Validate feature count
if X.shape[1] != 2048:
    st.error(
        f"❌ Expected **2 048 fingerprint columns** but found **{X.shape[1]}**. "
        "Please check your CSV."
    )
    st.stop()

# ── Predict ───────────────────────────────────────────────────────────────────
with st.spinner("🔬 Running predictions…"):
    pred_encoded = pipeline.predict(X)
    pred_labels  = [LABEL_MAP.get(int(p), str(p)) for p in pred_encoded]

    try:
        prob_matrix  = pipeline.predict_proba(X)           # shape (n, 3)
        confidence   = np.max(prob_matrix, axis=1) * 100   # % confidence
        has_proba    = True
    except AttributeError:
        confidence   = [None] * len(pred_labels)
        has_proba    = False

# ── Build results dataframe ───────────────────────────────────────────────────
results = pd.DataFrame()

if "smiles" in df_raw.columns:
    results["SMILES"] = df_raw["smiles"].values

results["Predicted Class"] = pred_labels

if has_proba:
    results["Confidence (%)"] = confidence.round(2)
    # Per-class probabilities
    for idx, name in LABEL_MAP.items():
        results[f"P({name}) (%)"] = (prob_matrix[:, idx] * 100).round(2)

if "label" in df_raw.columns:
    results["True Label"] = df_raw["label"].values
    results["Correct"] = results["True Label"].str.lower() == results["Predicted Class"].str.lower()

st.success(f"✅ Classified **{len(results)}** molecules")

st.divider()

# ── Metrics row ───────────────────────────────────────────────────────────────
counts = results["Predicted Class"].value_counts()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total molecules",  len(results))
col2.metric("🟢 Plastic",       counts.get("Plastic",        0))
col3.metric("🔴 Peptide",       counts.get("Peptide",         0))
col4.metric("🔵 Oligosaccharide", counts.get("Oligosaccharide", 0))

if has_proba:
    avg_conf = np.mean(confidence)
    st.metric("Average confidence", f"{avg_conf:.1f} %")

st.divider()

# ── Results table ─────────────────────────────────────────────────────────────
st.subheader("📊 Prediction Results")

# Colour "Correct" column if ground truth available
if "Correct" in results.columns:
    def _highlight(row):
        color = "background-color: #d4edda" if row["Correct"] else "background-color: #f8d7da"
        return [color] * len(row)
    st.dataframe(results.style.apply(_highlight, axis=1), use_container_width=True)
else:
    st.dataframe(results, use_container_width=True)

# ── Charts ────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Visualisations")

chart_col1, chart_col2 = st.columns(2)

# 1. Class distribution pie
with chart_col1:
    fig_pie = px.pie(
        names=counts.index,
        values=counts.values,
        color=counts.index,
        color_discrete_map=CLASS_COLORS,
        title="Predicted Class Distribution",
        hole=0.35,
    )
    fig_pie.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

# 2. Confidence histogram
if has_proba:
    with chart_col2:
        fig_hist = px.histogram(
            results,
            x="Confidence (%)",
            color="Predicted Class",
            color_discrete_map=CLASS_COLORS,
            nbins=40,
            title="Prediction Confidence Distribution",
            labels={"Confidence (%)": "Confidence (%)"},
        )
        fig_hist.update_layout(bargap=0.05)
        st.plotly_chart(fig_hist, use_container_width=True)

# 3. Per-class probability box plots
if has_proba:
    st.subheader("🎯 Per-class Probability Spread")
    prob_df = pd.DataFrame(
        prob_matrix * 100,
        columns=[LABEL_MAP[i] for i in range(3)]
    )
    prob_df["Predicted"] = pred_labels

    fig_box = px.box(
        prob_df.melt(id_vars="Predicted", var_name="Class", value_name="Probability (%)"),
        x="Class", y="Probability (%)", color="Class",
        color_discrete_map=CLASS_COLORS,
        title="Distribution of Predicted Probabilities per Class",
        points=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)

# 4. Accuracy (if ground truth present)
if "Correct" in results.columns:
    st.divider()
    acc = results["Correct"].mean() * 100
    st.metric("✅ Accuracy on uploaded data", f"{acc:.2f} %")

    conf_matrix_data = pd.crosstab(
        results["True Label"], results["Predicted Class"],
        rownames=["Actual"], colnames=["Predicted"]
    )
    fig_cm = px.imshow(
        conf_matrix_data,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        aspect="auto",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ── Download ──────────────────────────────────────────────────────────────────
st.divider()
csv_bytes = results.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Results CSV",
    data=csv_bytes,
    file_name="polymer_predictions.csv",
    mime="text/csv",
)
