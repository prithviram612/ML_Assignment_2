import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(page_title="Bank Marketing ML App", layout="wide")

st.title("📊 Bank Marketing Classification Dashboard")

# ---------------------------------
# Model Selection
# ---------------------------------
model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic_Regression",
        "Decision_Tree",
        "KNN",
        "Naive_Bayes",
        "Random_Forest",
        "XGBoost"
    ]
)

# ---------------------------------
# File Upload
# ---------------------------------
uploaded_file = st.file_uploader("Upload Bank CSV file", type=["csv"])

# ---------------------------------
# Main Logic
# ---------------------------------
if uploaded_file is not None:

    # Load dataset correctly (Bank dataset uses ;)
    df = pd.read_csv(uploaded_file, sep=';')
    df.columns = df.columns.str.strip()

    # Check if target column exists
    if "y" not in df.columns:
        st.error("Target column 'y' not found in uploaded dataset.")
        st.write("Detected columns:", df.columns)
        st.stop()

    # Encode categorical columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col])

    # Split features and target
    X = df.drop("y", axis=1)
    y = df["y"]

    # Load scaler
    scaler_path = "models/scaler.pkl"

    if not os.path.exists(scaler_path):
        st.error("Scaler file missing in models folder.")
        st.stop()

    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    # Load selected model
    model_path = f"models/{model_choice}.pkl"

    if not os.path.exists(model_path):
        st.error(f"Model file '{model_choice}.pkl' not found in models folder.")
        st.stop()

    model = joblib.load(model_path)

    # Predictions
    predictions = model.predict(X_scaled)

    # Some models may not support predict_proba
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_scaled)[:, 1]
    else:
        probabilities = None

    # ---------------------------------
    # Display Results
    # ---------------------------------
    st.subheader("📄 Classification Report")
    st.text(classification_report(y, predictions))

    # Confusion Matrix
    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y, predictions)

    fig1, ax1 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    st.pyplot(fig1)

    # ROC Curve (only if probabilities available)
    if probabilities is not None:
        st.subheader("📈 ROC Curve")
        fpr, tpr, _ = roc_curve(y, probabilities)
        roc_auc = auc(fpr, tpr)

        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax2.plot([0, 1], [0, 1], linestyle="--")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend()
        st.pyplot(fig2)

    st.success("✅ Model evaluation completed successfully!")
