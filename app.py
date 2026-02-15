import streamlit as st
import pandas as pd
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(page_title="Bank Marketing ML App", layout="wide")

st.title("📊 Bank Marketing Classification Dashboard")

# ---------------------------------
# Check dataset exists
# ---------------------------------
if not os.path.exists("bank.csv"):
    st.error("bank.csv not found in repository.")
    st.stop()

# ---------------------------------
# Load dataset
# ---------------------------------
df = pd.read_csv("bank.csv", sep=';')
df.columns = df.columns.str.strip()

if "y" not in df.columns:
    st.error("Target column 'y' not found in dataset.")
    st.stop()

# Encode categorical columns
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("y", axis=1)
y = df["y"]

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

run_button = st.button("OK")

# ---------------------------------
# Run Model Only When Button Clicked
# ---------------------------------
if run_button:

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
        st.error(f"Model file '{model_choice}.pkl' not found.")
        st.stop()

    model = joblib.load(model_path)

    # Predictions
    predictions = model.predict(X_scaled)

    # ---------------------------------
    # Classification Report as Table
    # ---------------------------------
    st.subheader("📄 Classification Report")

    report = classification_report(y, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

    # ---------------------------------
    # Confusion Matrix (Compact)
    # ---------------------------------
    st.subheader("📌 Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots(figsize=(2, 2))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap="Blues",
        cbar=False,
        annot_kws={"size": 8},
        ax=ax
    )

    plt.tight_layout()

    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
        st.pyplot(fig, use_container_width=False)

    st.success(f"✅ {model_choice} evaluation completed successfully!")
