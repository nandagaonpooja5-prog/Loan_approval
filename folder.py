import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    layout="centered"
)

# -------------------------------
# Title
# -------------------------------
st.title("Loan Approval Prediction System")
st.write("Using Decision Tree and Random Forest Algorithms")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("loan-prediction-dataset.csv")

# -------------------------------
# Handle Missing Values
# -------------------------------
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in ['LoanAmount', 'Loan_Amount_Term']:
    df[col] = df[col].fillna(df[col].median())

# Replace 3+ in Dependents
df['Dependents'] = df['Dependents'].replace('3+', 3)
df['Dependents'] = df['Dependents'].astype(int)

# -------------------------------
# Label Encoding
# -------------------------------
label_cols = [
    'Gender',
    'Married',
    'Education',
    'Self_Employed',
    'Loan_Status'
]

le = LabelEncoder()

for col in label_cols:
    df[col] = le.fit_transform(df[col])

# One Hot Encoding
df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)

# Convert Credit History
df['Credit_History'] = df['Credit_History'].astype(int)

# Drop Loan_ID if exists
if 'Loan_ID' in df.columns:
    df.drop('Loan_ID', axis=1, inplace=True)

# -------------------------------
# Features and Target
# -------------------------------
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# -------------------------------
# Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# Decision Tree Model
# -------------------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# -------------------------------
# Random Forest Model
# -------------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# -------------------------------
# Accuracy Scores
# -------------------------------
dt_accuracy = accuracy_score(
    y_test,
    dt_model.predict(X_test)
)

rf_accuracy = accuracy_score(
    y_test,
    rf_model.predict(X_test)
)

# -------------------------------
# Show Accuracy
# -------------------------------
st.subheader("Model Accuracy")

st.write(f"Decision Tree Accuracy: {dt_accuracy:.2f}")
st.write(f"Random Forest Accuracy: {rf_accuracy:.2f}")

# -------------------------------
# User Input Section
# -------------------------------
st.subheader("Enter Applicant Details")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.selectbox(
    "Married",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    [0, 1, 2, 3]
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0
)

loan_amount_term = st.number_input(
    "Loan Amount Term",
    min_value=0
)

credit_history = st.selectbox(
    "Credit History",
    [1, 0]
)

property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

# -------------------------------
# Encode User Input
# -------------------------------
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0

property_area_semiurban = 1 if property_area == "Semiurban" else 0
property_area_urban = 1 if property_area == "Urban" else 0

# -------------------------------
# Create Input DataFrame
# -------------------------------
input_data = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [credit_history],
    'Property_Area_Semiurban': [property_area_semiurban],
    'Property_Area_Urban': [property_area_urban]
})

# -------------------------------
# Model Selection
# -------------------------------
model_choice = st.radio(
    "Choose Model",
    ["Decision Tree", "Random Forest"]
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Loan Approval"):

    if model_choice == "Decision Tree":
        prediction = dt_model.predict(input_data)

    else:
        prediction = rf_model.predict(input_data)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")

# -------------------------------
# Feature Importance Analysis
# -------------------------------
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

st.subheader("Feature Importance for Loan Approval")

st.write(
    "Higher importance value means the feature has more impact on loan approval."
)

st.dataframe(feature_importance)

# Most Important Feature
top_feature = feature_importance.iloc[0]

st.success(
    f"Most Important Feature: {top_feature['Feature']} "
    f"with importance score {top_feature['Importance']:.3f}"
)

# -------------------------------
# Dataset Preview
# -------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())