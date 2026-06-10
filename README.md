# 🧪 Polymer Type Classification using Machine Learning

This project focuses on building a **machine learning model to classify polymers** into different categories based on their molecular structure.

The model predicts whether a molecule belongs to:

* Plastic
* Peptide
* Oligosaccharide

using **2048-bit Morgan fingerprint features** derived from molecular data.

---

## 📌 Project Overview

Polymer classification plays an important role in:

* Material science
* Drug discovery
* Chemical analysis

In this project:

* Molecular data is converted into numerical fingerprints
* A machine learning model is trained on this data
* The model learns patterns to classify polymers accurately

---

## 🧠 Approach

1. **Data Preprocessing**

   * Removed unnecessary columns (`smiles`, `label`, etc.)
   * Extracted 2048 fingerprint features

2. **Feature Representation**

   * Used **Morgan Fingerprints (ECFP)** for molecular encoding

3. **Model Training**

   * Algorithm: **XGBoost Classifier**
   * Trained on ~20,000+ samples

4. **Evaluation**

   * Achieved high accuracy (~100%) on test data

---

## 📊 Dataset

* Contains molecular data with:

  * SMILES (optional)
  * Fingerprint features (0–2047)
  * Labels (polymer types)
https://www.kaggle.com/datasets/victorsabanzagil/polymers/data
⚠️ For prediction, only fingerprint features are required.

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost

---

## 📈 Model Details

* Model: XGBoost Classifier
* Input: 2048-bit fingerprint vector
* Output: Polymer class
* Performance: ~100% accuracy on test set

---

## 🌐 Deployment (Streamlit)

The trained model is deployed using **Streamlit** to allow users to upload data and get predictions interactively.
https://mysapftthkhti9qwmb8rxf.streamlit.app/

### 🔹 Features of the App

* Upload CSV file for batch prediction
* Predict polymer type for each molecule
* Display:

  * Predicted class
  * Confidence score
  * Class probabilities
* Interactive visualizations

---



## 📤 Input Format

CSV file should contain:

```
0,1,2,...,2047
```

(Optional columns like `smiles`, `label` are ignored)

---
<img width="1884" height="989" alt="image" src="https://github.com/user-attachments/assets/e473cd07-763e-4cb8-91c2-1d5871aece6a" />

<img width="1915" height="892" alt="image" src="https://github.com/user-attachments/assets/c5e83068-63b2-44fe-ba4f-b70d53a72f53" />


