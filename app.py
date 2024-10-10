# app.py

import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import os

# Import custom functions from model_utils.py
from model_utils import (
    load_model_from_github,
    load_scaler_from_github,
    load_training_columns_from_github,
    load_dataset_from_github
)

# URLs to resources

MODEL_URL = 'https://github.com/tadiwamark/AML/releases/download/gbc/gbc_aml_model.pkl'
SCALER_URL = 'https://github.com/tadiwamark/AML/releases/download/gbc/scaler_gb.save'
COLUMNS_URL = 'https://github.com/tadiwamark/AML/releases/download/gbc_training_columns/training_columns.pkl'
DATASET_URL = 'https://github.com/tadiwamark/AML/blob/main/HI-Small_Trans.csv'

# Load the model, scaler, training columns, and dataset
@st.cache_resource
def load_resources():
    gbc = load_model_from_github(MODEL_URL)
    scaler = load_scaler_from_github(SCALER_URL)
    training_columns = load_training_columns_from_github(COLUMNS_URL)
    df = load_dataset_from_github(DATASET_URL)
    return gbc, scaler, training_columns, df

gbc, scaler, training_columns, df = load_resources()

# Preprocess the data
def preprocess_data(df, scaler, training_columns):
    df = df.drop_duplicates()
    df = df.dropna()

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df.drop(['Timestamp', 'From Bank', 'To Bank', 'Account', 'Account.1', 'Receiving Currency', 'Amount Received'], axis=1, inplace=True)

    X = df.drop('Is Laundering', axis=1)
    y = df['Is Laundering']

    # One-hot encoding
    X = pd.get_dummies(X, columns=['Payment Currency', 'Payment Format'], drop_first=True)

    # Align columns with the trained model
    missing_cols = set(training_columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    X = X[training_columns]

    # Scaling
    numeric_cols = ['Amount Paid', 'Hour']
    X[numeric_cols] = scaler.transform(X[numeric_cols])

    return X, y

X, y = preprocess_data(df, scaler, training_columns)

# Function to simulate real-time data
def get_live_data():
    idx = random.randint(0, len(X) - 1)
    transaction = X.iloc[idx]
    actual_label = y.iloc[idx]
    return transaction.to_frame().T, actual_label

# Streamlit app
st.title("Real-Time Transaction Classification with GradientBoostingClassifier")

st.markdown("""
This application simulates real-time transaction classification using a GradientBoostingClassifier model.
Transactions are streamed in real-time and classified as normal or fraudulent.
""")

placeholder = st.empty()

for i in range(1, 101):
    transaction, actual_label = get_live_data()
    prediction_prob = gbc.predict_proba(transaction)[0][1]
    prediction = gbc.predict(transaction)[0]
    result = 'Fraudulent Transaction Detected' if prediction == 1 else 'Normal Transaction'
    with placeholder.container():
        st.write(f"**Transaction {i}:**")
        st.write(f"**Result:** {result}")
        st.write(f"**Actual Label:** {'Fraud' if actual_label == 1 else 'Normal'}")
        st.write(f"**Prediction Probability:** {prediction_prob:.6f}")
        st.write("---")
    time.sleep(1)
