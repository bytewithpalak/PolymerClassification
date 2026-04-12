import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Page config
st.set_page_config(page_title="Polymer Classifier", page_icon="🧪")

st.title("🧪 Polymer Classification App")
st.caption("Upload fingerprint dataset (CSV) for prediction")

# Load combined pipeline (NO scaler/encoder separately)
model = joblib.load("model.pkl")

st.divider()

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("📄 Preview of Uploaded Data:")
    st.dataframe(df.head())

    # Drop unwanted columns safely
    drop_cols = []
    for col in ["label", "smiles", "Unnamed: 0"]:
        if col in df.columns:
            drop_cols.append(col)

    X = df.drop(columns=drop_cols)

    # Check feature count (important)
    if X.shape[1] != 2048:
        st.error(f"❌ Expected 2048 feature columns. Found {X.shape[1]}.")
    else:
        # Prediction using pipeline
        predictions = model.predict(X)

        # If model supports probabilities
        try:
            probabilities = model.predict_proba(X)
            confidence = np.max(probabilities, axis=1) * 100
        except:
            confidence = [None] * len(predictions)

        # Add results to dataframe
        df["Predicted_Class"] = predictions
        df["Confidence (%)"] = confidence

        st.success("✅ Prediction Complete")

        st.write("📊 Results:")
        st.dataframe(df.head())

        # ---------- CHART ----------
        st.subheader("📈 Predicted Class Distribution")

        fig = px.histogram(
            df,
            x="Predicted_Class",
            color="Predicted_Class",
            title="Predicted Class Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)
