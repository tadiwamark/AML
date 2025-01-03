import pandas as pd
import numpy as np
import streamlit as st
import joblib
import random
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- Load Dataset ---
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# --- Data Preprocessing ---
def preprocess_data(df):
    df = df.dropna()  # Remove rows with missing values
    # Select relevant columns for model training
    relevant_cols = [
        'From Bank', 'To Bank', 'Amount Received', 'Amount Paid',
        'Payment Format'
    ]
    df = df[relevant_cols]

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['Payment Format'], drop_first=True)

    # Normalize numeric columns
    numeric_cols = ['From Bank', 'To Bank', 'Amount Received', 'Amount Paid']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, scaler

# --- Model Training ---
def train_model(df):
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    model.fit(df)
    return model

# --- Simulated Transactions ---
def generate_transactions(n, fraud_percent, scaler):
    legitimate = {
        'From Bank': np.random.randint(1000, 2000, int(n * (1 - fraud_percent))),
        'To Bank': np.random.randint(3000, 4000, int(n * (1 - fraud_percent))),
        'Amount Received': np.random.normal(loc=500, scale=50, size=int(n * (1 - fraud_percent))),
        'Amount Paid': np.random.normal(loc=500, scale=50, size=int(n * (1 - fraud_percent))),
        'Payment Format': ['credit'] * int(n * (1 - fraud_percent))
    }
    fraudulent = {
        'From Bank': np.random.randint(1000, 2000, int(n * fraud_percent)),
        'To Bank': np.random.randint(3000, 4000, int(n * fraud_percent)),
        'Amount Received': np.random.normal(loc=5000, scale=500, size=int(n * fraud_percent)),
        'Amount Paid': np.random.normal(loc=5000, scale=500, size=int(n * fraud_percent)),
        'Payment Format': ['wire'] * int(n * fraud_percent)
    }

    legitimate_df = pd.DataFrame(legitimate)
    fraudulent_df = pd.DataFrame(fraudulent)
    transactions = pd.concat([legitimate_df, fraudulent_df]).sample(frac=1).reset_index(drop=True)

    # One-hot encode and scale
    transactions = pd.get_dummies(transactions, columns=['Payment Format'], drop_first=True)
    numeric_cols = ['From Bank', 'To Bank', 'Amount Received', 'Amount Paid']
    transactions[numeric_cols] = scaler.transform(transactions[numeric_cols])

    return transactions

# --- Streamlit App ---
st.title('Real-Time Anti-Money Laundering Detection')

uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=['csv'])
if uploaded_file:
    st.write("Loading dataset...")
    dataset = load_data(uploaded_file)
    st.write("Dataset Loaded:", dataset.head())

    st.write("Preprocessing data...")
    preprocessed_data, scaler = preprocess_data(dataset)
    st.write("Data Preprocessed Successfully.")

    st.write("Training model...")
    model = train_model(preprocessed_data)
    joblib.dump(model, "aml_model.pkl")
    st.write("Model Trained and Saved.")

    st.sidebar.title("Simulation Settings")
    fraud_percentage = st.sidebar.slider("Percentage of Fraudulent Transactions", 0.0, 1.0, 0.1, 0.01)

    if st.sidebar.button("Start Simulation"):
        st.write("Generating Transactions...")
        transactions = generate_transactions(60, fraud_percentage, scaler)
        st.write("Transactions Generated:", transactions.head())

        st.write("Running Model...")
        predictions = model.predict(transactions)
        transactions['Anomaly'] = np.where(predictions == -1, 1, 0)

        flagged_transactions = transactions[transactions['Anomaly'] == 1]
        st.write("Flagged Transactions:", flagged_transactions)

        st.write("Simulation Complete.")

