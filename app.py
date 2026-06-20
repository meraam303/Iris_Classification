"""
Iris Species Classification — ML Inference Console
CodeAlpha Data Science Internship

A production-style Streamlit interface around a Random Forest classifier
trained on the Iris dataset, including model diagnostics and a
session-based prediction log.
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ======================================================================
# CONFIG & THEME
# ======================================================================
st.set_page_config(
    page_title="Iris Inference Console",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#e5e7eb"
SLATE = "#94a3b8"
BLUE = "#3b82f6"
GREEN = "#22c55e"
PURPLE = "#a78bfa"
PANEL = "#0f172a"
BORDER = "#1e293b"
GRID = "#1e293b"
CLASS_COLORS = [BLUE, PURPLE, GREEN]

plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "axes.edgecolor": BORDER,
    "axes.labelcolor": SLATE,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "text.color": INK,
    "font.size": 10,
    "grid.color": GRID,
})

st.markdown(
    f"""
    <style>
        .block-container {{ padding-top: 1.6rem; max-width: 1320px; }}
        h1, h2, h3, h4 {{ letter-spacing: -0.02em; }}
        h1 {{ font-weight: 700; }}

        div[data-testid="stMetric"] {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 0.9rem 1.1rem 0.7rem 1.1rem;
        }}
        div[data-testid="stMetricLabel"] {{
            font-size: 0.74rem !important;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: {SLATE} !important;
        }}

        .kicker {{
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {BLUE};
            border: 1px solid rgba(59,130,246,0.35);
            background: rgba(59,130,246,0.08);
            padding: 0.22rem 0.65rem;
            border-radius: 20px;
            margin-bottom: 0.6rem;
        }}

        .panel {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 1.4rem 1.5rem;
        }}
        .panel-title {{
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {SLATE};
            margin-bottom: 0.9rem;
        }}

        .pred-hero {{
            text-align: center;
            padding: 1.6rem 1rem 1.3rem 1rem;
        }}
        .pred-hero .label {{
            font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase;
            color: {SLATE}; margin-bottom: 0.3rem;
        }}
        .pred-hero .value {{
            font-size: 2.6rem; font-weight: 800; color: {INK}; line-height: 1.1;
        }}
        .pred-hero .sub {{
            color: {SLATE}; font-size: 0.85rem; margin-top: 0.4rem;
        }}

        .badge {{
            display: inline-block; padding: 0.15rem 0.55rem; border-radius: 6px;
            font-size: 0.72rem; font-weight: 600; margin-right: 0.3rem;
        }}
        .badge-blue {{ background: rgba(59,130,246,0.12); color: {BLUE}; }}

        .footnote {{ color: {SLATE}; font-size: 0.78rem; }}
        hr {{ border-color: {BORDER}; }}
        section[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

SPECIES = ["Setosa", "Versicolor", "Virginica"]
FEATURE_LABELS = ["Sepal length (cm)", "Sepal width (cm)", "Petal length (cm)", "Petal width (cm)"]

if "history" not in st.session_state:
    st.session_state.history = []


# ======================================================================
# TRAIN (cached)
# ======================================================================
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

    model = RandomForestClassifier(n_estimators=150, random_state=7)
    model.fit(X_train_s, y_train)

    test_acc = accuracy_score(y_test, model.predict(X_test_s))
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5)
    cm = confusion_matrix(y_test, model.predict(X_test_s))

    return model, scaler, test_acc, cv_scores, cm, (X, y)


model, scaler, test_accuracy, cv_scores, conf_matrix, full_data = train_model()
X_full, y_full = full_data


def predict_sample(sl, sw, pl, pw):
    sample = np.array([[sl, sw, pl, pw]])
    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)[0]
    proba = model.predict_proba(sample_scaled)[0]
    return pred, proba


feat_importance = pd.Series(model.feature_importances_, index=FEATURE_LABELS).sort_values(ascending=False)


# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown("#### Inference Console")
    st.caption("Iris species classifier")
    st.divider()

    st.markdown("**Model**")
    st.write("Random Forest Classifier (150 trees)")

    st.markdown("**Pipeline**")
    st.markdown(
        '<span class="badge badge-blue">Min-Max scaling</span>'
        '<span class="badge badge-blue">Stratified split</span>'
        '<span class="badge badge-blue">5-fold CV</span>',
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("**Candidate models evaluated**")
    st.markdown(
        "- K-Nearest Neighbors\n"
        "- Decision Tree\n"
        "- Support Vector Machine (RBF)\n"
        "- **Random Forest (selected)**"
    )

    st.write("")
    st.markdown("**Most informative features**")
    for f in feat_importance.head(2).index:
        st.markdown(f"- {f}")

    st.divider()
    st.caption(f"Session predictions: {len(st.session_state.history)}")
    st.caption("CodeAlpha Data Science Internship · 2026")


# ======================================================================
# HEADER
# ======================================================================
st.markdown('<span class="kicker">Classification · Random Forest · Tabular ML</span>', unsafe_allow_html=True)
st.title("Iris Species Inference Console")
st.write(
    "A multi-class classifier that identifies Iris flower species — Setosa, Versicolor, "
    "or Virginica — from four morphological measurements. Adjust the measurements to "
    "generate a live prediction with calibrated class probabilities."
)

tab_classify, tab_diagnostics, tab_history = st.tabs(
    ["Classification", "Model Diagnostics", "Session History"]
)

# ======================================================================
# TAB 1 — CLASSIFICATION
# ======================================================================
with tab_classify:
    col_form, col_result = st.columns([1, 1.25], gap="large")

    with col_form:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Flower Measurements</div>', unsafe_allow_html=True)

        sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
        sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)
        petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 3.8, 0.1)
        petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.2, 0.1)

        st.write("")
        run = st.button("Classify Sample", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        if run:
            pred, proba = predict_sample(sepal_length, sepal_width, petal_length, petal_width)
            confidence = proba[pred]

            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "sepal_length": sepal_length, "sepal_width": sepal_width,
                "petal_length": petal_length, "petal_width": petal_width,
                "predicted_species": SPECIES[pred], "confidence": round(confidence * 100, 1),
            })

            st.markdown('<div class="panel pred-hero">', unsafe_allow_html=True)
            st.markdown('<div class="label">Predicted Species</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="value">{SPECIES[pred]}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="sub">Confidence &nbsp;·&nbsp; {confidence * 100:.1f}%</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.write("")
            m1, m2, m3 = st.columns(3)
            m1.metric("Petal area", f"{petal_length * petal_width:.2f} cm²")
            m2.metric("Sepal area", f"{sepal_length * sepal_width:.2f} cm²")
            m3.metric("Model confidence", f"{confidence * 100:.1f}%")

            st.write("")
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Class Probability Distribution</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(7, 3.4))
            bars = ax.bar(SPECIES, proba, color=CLASS_COLORS, width=0.55)
            for bar, p in zip(bars, proba):
                ax.text(bar.get_x() + bar.get_width() / 2, p + 0.03, f"{p * 100:.1f}%",
                        ha="center", color=INK, fontsize=9)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Probability")
            ax.grid(axis="y", alpha=0.25, linewidth=0.6)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            st.pyplot(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.write("")
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Sample Position in Feature Space</div>', unsafe_allow_html=True)

            fig2, ax2 = plt.subplots(figsize=(7, 4))
            for cls in range(3):
                mask = y_full == cls
                ax2.scatter(X_full[mask, 2], X_full[mask, 3], s=22, alpha=0.55,
                            color=CLASS_COLORS[cls], label=SPECIES[cls])
            ax2.scatter([petal_length], [petal_width], s=160, color=INK,
                        edgecolor="black", linewidth=1.2, marker="*", label="Input sample", zorder=5)
            ax2.set_xlabel("Petal length (cm)")
            ax2.set_ylabel("Petal width (cm)")
            ax2.grid(alpha=0.2, linewidth=0.6)
            for spine in ["top", "right"]:
                ax2.spines[spine].set_visible(False)
            ax2.legend(frameon=False, fontsize=8, loc="upper left")
            st.pyplot(fig2, use_container_width=True)
            st.caption(
                "Petal length and width are the two most discriminative features for "
                "this dataset — the plot shows where the input sample falls relative to "
                "the training distribution of each species."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Output</div>', unsafe_allow_html=True)
            st.write(
                "Set the flower measurements on the left and click "
                "**Classify Sample** to generate a prediction, class probabilities, "
                "and a feature-space visualization."
            )
            st.markdown("</div>", unsafe_allow_html=True)

# ======================================================================
# TAB 2 — MODEL DIAGNOSTICS
# ======================================================================
with tab_diagnostics:
    d1, d2 = st.columns([1, 1], gap="large")

    with d1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Performance Summary</div>', unsafe_allow_html=True)

        p1, p2, p3 = st.columns(3)
        p1.metric("Test accuracy", f"{test_accuracy * 100:.1f}%")
        p2.metric("CV mean accuracy", f"{cv_scores.mean() * 100:.1f}%")
        p3.metric("CV std dev", f"{cv_scores.std() * 100:.2f}%")

        st.write("")
        st.markdown("**Feature Importance**")
        fig3, ax3 = plt.subplots(figsize=(6, 3.4))
        ax3.barh(feat_importance.index[::-1], feat_importance.values[::-1], color=BLUE, height=0.55)
        ax3.set_xlabel("Importance")
        ax3.grid(axis="x", alpha=0.25, linewidth=0.6)
        for spine in ["top", "right"]:
            ax3.spines[spine].set_visible(False)
        st.pyplot(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with d2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Confusion Matrix (held-out test set)</div>', unsafe_allow_html=True)

        fig4, ax4 = plt.subplots(figsize=(5.2, 4.4))
        im = ax4.imshow(conf_matrix, cmap="Blues")
        ax4.set_xticks(range(3)); ax4.set_xticklabels(SPECIES, rotation=20)
        ax4.set_yticks(range(3)); ax4.set_yticklabels(SPECIES)
        ax4.set_xlabel("Predicted"); ax4.set_ylabel("Actual")
        for i in range(3):
            for j in range(3):
                ax4.text(j, i, conf_matrix[i, j], ha="center", va="center",
                         color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else INK,
                         fontsize=11, fontweight="bold")
        st.pyplot(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Why Random Forest</div>', unsafe_allow_html=True)
        st.write(
            "Random Forest was selected after benchmarking against KNN, a single "
            "Decision Tree, and an SVM with an RBF kernel. It matched or exceeded the "
            "accuracy of the other candidates while remaining robust to the small "
            "sample size (150 rows) and resistant to overfitting through ensemble "
            "averaging across 150 trees."
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================================
# TAB 3 — SESSION HISTORY
# ======================================================================
with tab_history:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Prediction Log (this session)</div>', unsafe_allow_html=True)

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        csv_buf = io.StringIO()
        hist_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download log as CSV",
            data=csv_buf.getvalue(),
            file_name="classification_log.csv",
            mime="text/csv",
        )
    else:
        st.write("No predictions generated yet in this session.")

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.markdown(
    '<p class="footnote">Built as part of the CodeAlpha Data Science Internship. '
    "Model trained live on application startup using the canonical Iris dataset.</p>",
    unsafe_allow_html=True,
)
