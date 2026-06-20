"""
Iris Flower Classification — Interactive Streamlit App
CodeAlpha Data Science Internship
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
)

SPECIES = ["Setosa", "Versicolor", "Virginica"]
COLORS = ["#f472b6", "#818cf8", "#34d399"]
FEATURES = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]


# ---------------------------------------------------------
# Train (cached so it only runs once per session)
# ---------------------------------------------------------
@st.cache_resource
def train_model():
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=7)
    model.fit(X_train_s, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_s))
    return model, scaler, acc


model, scaler, test_accuracy = train_model()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🌸 Iris Flower Classification")
st.markdown(
    "Predict the species of an Iris flower (**Setosa**, **Versicolor**, **Virginica**) "
    "from its sepal and petal measurements using a Random Forest model."
)
st.info(f"Model test accuracy: **{test_accuracy * 100:.2f}%**")

st.divider()

# ---------------------------------------------------------
# Layout: input sliders + prediction
# ---------------------------------------------------------
col_input, col_result = st.columns([1, 1.3])

with col_input:
    st.subheader("🔧 Flower Measurements (cm)")
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8, 0.1)
    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0, 0.1)
    petal_length = st.slider("Petal Length", 1.0, 7.0, 3.8, 0.1)
    petal_width = st.slider("Petal Width", 0.1, 2.5, 1.2, 0.1)

    predict_btn = st.button("🔍 Predict Species", type="primary", use_container_width=True)

with col_result:
    st.subheader("🎯 Prediction")

    if predict_btn:
        sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        sample_scaled = scaler.transform(sample)

        pred = model.predict(sample_scaled)[0]
        proba = model.predict_proba(sample_scaled)[0]

        st.success(f"Predicted species: **{SPECIES[pred]}**")

        proba_df = pd.DataFrame({"Species": SPECIES, "Probability": proba})

        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(proba_df["Species"], proba_df["Probability"], color=COLORS, alpha=0.9, edgecolor="black")
        for bar, p in zip(bars, proba):
            ax.text(bar.get_x() + bar.get_width() / 2, p + 0.02, f"{p*100:.1f}%", ha="center")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Probability")
        ax.set_title("Class Probabilities")
        st.pyplot(fig)
    else:
        st.write("👈 Set the measurements and click **Predict Species**.")

st.divider()

# ---------------------------------------------------------
# Footer / About
# ---------------------------------------------------------
with st.expander("ℹ️ About this project"):
    st.markdown(
        """
        - **Dataset**: Iris dataset (150 samples, 3 balanced classes)
        - **Preprocessing**: Missing-value imputation, Label Encoding, Min-Max Scaling
        - **Models compared**: KNN, Decision Tree, Random Forest, SVM (RBF)
        - **Final model**: Random Forest Classifier
        - Built as part of the **CodeAlpha Data Science Internship**
        """
    )
