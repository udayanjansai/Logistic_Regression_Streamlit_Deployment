import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("Customer Churn Prediction (Logistic Regression)")

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = load_data()

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)   

load_css("style.css")  
# --------------------------------------------------
# Dataset preview
# --------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# --------------------------------------------------
# Data preprocessing
# --------------------------------------------------
df = df.drop(columns=["customerID"])

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})

X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
X["TotalCharges"].fillna(X["TotalCharges"].median(), inplace=True)

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

# --------------------------------------------------
# Train–test split (FIRST)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------
# One-Hot Encoding (FIT ONLY ON TRAIN)
# --------------------------------------------------
encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

X_train_cat = encoder.fit_transform(X_train[cat_cols])
X_test_cat = encoder.transform(X_test[cat_cols])

X_train_cat_df = pd.DataFrame(
    X_train_cat,
    columns=encoder.get_feature_names_out(cat_cols),
    index=X_train.index
)

X_test_cat_df = pd.DataFrame(
    X_test_cat,
    columns=encoder.get_feature_names_out(cat_cols),
    index=X_test.index
)

# --------------------------------------------------
# Scale numeric columns (FIT ONLY ON TRAIN)
# --------------------------------------------------
sc = StandardScaler()

X_train_num = pd.DataFrame(
    sc.fit_transform(X_train[num_cols]),
    columns=num_cols,
    index=X_train.index
)

X_test_num = pd.DataFrame(
    sc.transform(X_test[num_cols]),
    columns=num_cols,
    index=X_test.index
)

# --------------------------------------------------
# Final training/testing data
# --------------------------------------------------
X_train_final = pd.concat([X_train_num, X_train_cat_df], axis=1)
X_test_final = pd.concat([X_test_num, X_test_cat_df], axis=1)

# 🔑 SAVE FEATURE ORDER
feature_columns = X_train_final.columns

# --------------------------------------------------
# Train Logistic Regression
# --------------------------------------------------
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_final, y_train)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
st.subheader(" Model Evaluation")

y_pred = model.predict(X_test_final)
accuracy = accuracy_score(y_test, y_pred)

st.metric("Accuracy", f"{accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, cmap="cividis", values_format="d")
st.pyplot(fig)

# --------------------------------------------------
# Prediction UI
# --------------------------------------------------
st.subheader(" Predict Customer Churn")

user_input = {}

for col in X.columns:
    if col in cat_cols:
        user_input[col] = st.selectbox(col, sorted(X[col].unique()))
    elif col == "SeniorCitizen":
        user_input[col] = st.selectbox(col, [0, 1])

    else:
        user_input[col] = st.number_input(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )

input_df = pd.DataFrame([user_input])

# ----- transform numeric -----
input_num_scaled = pd.DataFrame(
    sc.transform(input_df[num_cols]),
    columns=num_cols
)

# ----- transform categorical -----
input_cat = encoder.transform(input_df[cat_cols])
input_cat_df = pd.DataFrame(
    input_cat,
    columns=encoder.get_feature_names_out(cat_cols)
)

# ----- combine -----
input_final = pd.concat(
    [input_num_scaled.reset_index(drop=True),
     input_cat_df.reset_index(drop=True)],
    axis=1
)

# 🔑 ALIGN WITH TRAINING FEATURES
input_final = input_final.reindex(columns=feature_columns, fill_value=0)

# ----- predict -----
prediction = model.predict(input_final)[0]
probability = model.predict_proba(input_final)[0][1]

if prediction == 1:
    st.error(f" Customer is likely to CHURN (Probability: {probability:.2f})")
else:
    st.success(f" Customer is NOT likely to churn (Probability: {probability:.2f})")
