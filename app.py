import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Load CSS
# --------------------------------------------------
def load_css(file):
    if os.path.exists(file):
        with open(file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<div class="header">
    <h1>Telco Customer Churn Prediction</h1>
    <p>Logistic Regression | Binary Classification</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.drop("customerID", axis=1, inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = pd.get_dummies(df, drop_first=True)
    return df

df = load_data()

# --------------------------------------------------
# Sidebar Summary
# --------------------------------------------------
st.sidebar.markdown("## Dataset Overview")
st.sidebar.write("Rows:", df.shape[0])
st.sidebar.write("Features:", df.shape[1] - 1)
st.sidebar.write("Churn Rate:", f"{df['Churn'].mean()*100:.2f}%")

# --------------------------------------------------
# Features & Target
# --------------------------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# --------------------------------------------------
# Train Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------
# Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# Model Training
# --------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --------------------------------------------------
# Predictions
# --------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

# --------------------------------------------------
# Metrics Section
# --------------------------------------------------
st.markdown("## Model Performance")

m1, m2, m3 = st.columns(3)
m1.metric("Accuracy", f"{accuracy*100:.2f}%")
m2.metric("ROC-AUC Score", f"{auc:.3f}")
m3.metric("Total Customers", len(df))

# --------------------------------------------------
# Confusion Matrix & ROC
# --------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig)

with c2:
    st.markdown("### ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    st.pyplot(fig)

# --------------------------------------------------
# Classification Report
# --------------------------------------------------
st.markdown("## Classification Report")
st.code(classification_report(y_test, y_pred))

# --------------------------------------------------
# Churn Distribution
# --------------------------------------------------
st.markdown("## Customer Distribution")

stay = (y == 0).sum()
leave = (y == 1).sum()

d1, d2 = st.columns(2)
d1.metric("Customers Staying", stay)
d2.metric("Customers Leaving", leave)

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.markdown("## Predict New Customer Churn")

tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly = st.slider("Monthly Charges", 0.0, 200.0, 70.0)
total = st.slider("Total Charges", 0.0, 10000.0, 1000.0)

sample = np.zeros(X.shape[1])
sample[X.columns.get_loc("tenure")] = tenure
sample[X.columns.get_loc("MonthlyCharges")] = monthly
sample[X.columns.get_loc("TotalCharges")] = total

sample_scaled = scaler.transform(sample.reshape(1, -1))
pred = model.predict(sample_scaled)[0]
prob = model.predict_proba(sample_scaled)[0][1]

if pred == 1:
    st.error(f"Customer Likely to LEAVE (Probability: {prob:.2f})")
else:
    st.success(f"Customer Likely to STAY (Probability: {1-prob:.2f})")
