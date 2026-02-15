import streamlit as st
import pandas as pd
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

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

run_button = st.button("OK")

# ---------------------------------
# Run Model After Button Click
# ---------------------------------
if run_button:

    scaler_path = "models/scaler.pkl"

    if not os.path.exists(scaler_path):
        st.error("Scaler file missing.")
        st.stop()

    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    model_path = f"models/{model_choice}.pkl"

    if not os.path.exists(model_path):
        st.error(f"{model_choice}.pkl not found.")
        st.stop()

    model = joblib.load(model_path)

    predictions = model.predict(X_scaled)

    st.subheader("📄 Classification Report")

    report = classification_report(y, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

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

    st.success(f"{model_choice} evaluation completed.")
