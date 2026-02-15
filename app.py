import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

st.title("Bank Marketing Classification App")

# --------------------
# Upload Dataset
# --------------------
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression",
     "Decision Tree",
     "KNN",
     "Naive Bayes",
     "Random Forest",
     "XGBoost"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    model = joblib.load(f"models/{model_choice}.pkl")

    X = df.drop("y", axis=1)
    y = df["y"]

    predictions = model.predict(X)

    st.subheader("Classification Report")
    st.text(classification_report(y, predictions))

    cm = confusion_matrix(y, predictions)

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig)
