import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load combined pipeline model
model = joblib.load("model.pkl")

# Page config
st.set_page_config(page_title="Polymer Classifier", layout="wide")

# Title
st.title("🧪 Polymer Classification App")
st.write("Predict the type of polymer based on input features")

# ---------- SIDEBAR INPUT ----------
st.sidebar.header("Enter Input Features")

# 👉 Change these according to YOUR dataset features
f1 = st.sidebar.number_input("Feature 1", value=0.0)
f2 = st.sidebar.number_input("Feature 2", value=0.0)
f3 = st.sidebar.number_input("Feature 3", value=0.0)
f4 = st.sidebar.number_input("Feature 4", value=0.0)

# ---------- PREDICTION ----------
if st.sidebar.button("Predict"):

    # Input array
    input_data = np.array([[f1, f2, f3, f4]])

    # Prediction
    prediction = model.predict(input_data)[0]

    # ---------- OUTPUT ----------
    st.subheader("📊 Prediction Result")

    st.metric("Predicted Polymer Type", prediction)

    # ---------- OPTIONAL CHART ----------
    st.subheader("📈 Input Feature Overview")

    features = ["F1", "F2", "F3", "F4"]
    values = [f1, f2, f3, f4]

    fig, ax = plt.subplots()
    ax.bar(features, values)
    ax.set_title("Input Features")
    ax.set_ylabel("Values")

    st.pyplot(fig)
    