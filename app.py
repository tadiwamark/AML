import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import requests
import os
import time
import random
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="Real-Time AML Transaction Monitor", layout="wide")

# Download and load the model
@st.cache_resource
def load_model():
    model_url = "https://github.com/tadiwamark/AML/releases/download/dnn_aml/dnn_aml_model.h5"  # Replace with your URL
    model_path = "dnn_aml_model.h5"

    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            response = requests.get(model_url, stream=True)
            if response.status_code == 200:
                with open(model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
            else:
                st.error("Failed to download model. Check the URL.")
                return None

    return tf.keras.models.load_model(model_path)

# Preprocess data
def preprocess_data(data, scaler):
    # Convert 'Timestamp' to datetime and extract features
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Year'] = data['Timestamp'].dt.year
    data['Month'] = data['Timestamp'].dt.month
    data['Day'] = data['Timestamp'].dt.day
    data['Hour'] = data['Timestamp'].dt.hour
    data['Minute'] = data['Timestamp'].dt.minute
    data = data.drop(columns=['Timestamp'])

    # Encode categorical variables
    categorical_columns = ['Receiving Currency', 'Payment Currency', 'Payment Format']
    for col in categorical_columns:
        data[col] = data[col].map({'USD': 0, 'EUR': 1, 'GBP': 2, 'Wire': 0, 'Credit Card': 1, 'Cheque': 2, 'Reinvestment': 3})

    # Encode account columns using hash encoding
    data['Account'] = data['Account'].astype(str).apply(lambda x: hash(x) % (10**6))
    data['Account.1'] = data['Account.1'].astype(str).apply(lambda x: hash(x) % (10**6))

    # Scale numerical columns
    numeric_columns = ['Amount Received', 'Amount Paid']
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    return data

# Generate simulated transactions
def generate_transactions(num_transactions, anomaly_rate=0.1):
    np.random.seed(int(time.time()))
    transactions = pd.DataFrame({
        'Timestamp': pd.date_range(start=pd.Timestamp.now(), periods=num_transactions, freq='T'),
        'From Bank': np.random.randint(1, 1000, num_transactions),
        'Account': [f'8000{np.random.randint(1000, 9999)}' for _ in range(num_transactions)],
        'To Bank': np.random.randint(1, 1000, num_transactions),
        'Account.1': [f'8000{np.random.randint(1000, 9999)}' for _ in range(num_transactions)],
        'Amount Received': np.random.uniform(0.01, 10000, num_transactions),
        'Receiving Currency': np.random.choice(['USD', 'EUR', 'GBP'], num_transactions),
        'Amount Paid': np.random.uniform(0.01, 10000, num_transactions),
        'Payment Currency': np.random.choice(['USD', 'EUR', 'GBP'], num_transactions),
        'Payment Format': np.random.choice(['Wire', 'Credit Card', 'Cheque', 'Reinvestment'], num_transactions)
    })

    # Introduce anomalies
    num_anomalies = int(num_transactions * anomaly_rate)
    if num_anomalies > 0:
        anomaly_indices = random.sample(range(num_transactions), num_anomalies)
        for idx in anomaly_indices:
            transactions.loc[idx, 'Amount Received'] = np.random.uniform(10000, 50000)  # Unusually high amounts
            transactions.loc[idx, 'From Bank'] = transactions.loc[idx, 'To Bank']  # Same source and destination

    return transactions

# Initialize resources
model = load_model()
scaler = StandardScaler()

# Streamlit UI
st.title("💸 Real-Time Anti-Money Laundering (AML) Monitor")
st.sidebar.title("Settings")
st.sidebar.markdown("Configure transaction simulation settings.")

# Simulation settings
batch_size = st.sidebar.slider("Batch Size", min_value=10, max_value=100, value=60, step=10)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", min_value=1, max_value=10, value=5, step=1)
anomaly_rate = st.sidebar.slider("Anomaly Rate (%)", min_value=0, max_value=50, value=10, step=5) / 100

# Data storage
if "transactions" not in st.session_state:
    st.session_state["transactions"] = pd.DataFrame()
if "flagged" not in st.session_state:
    st.session_state["flagged"] = pd.DataFrame()

# Simulation and processing
with st.container():
    st.header("📊 Real-Time Transactions")
    st.write("Simulating and processing transactions in real-time to detect suspicious activity.")

    transactions = generate_transactions(batch_size, anomaly_rate)
    preprocessed_data = preprocess_data(transactions, scaler)

    # Predict using the loaded model
    predictions = (model.predict(preprocessed_data) > 0.5).astype(int)
    transactions['Is Laundering'] = predictions

    # Append new transactions to session state
    st.session_state["transactions"] = pd.concat([st.session_state["transactions"], transactions])
    flagged_transactions = transactions[transactions['Is Laundering'] == 1]
    st.session_state["flagged"] = pd.concat([st.session_state["flagged"], flagged_transactions])

    # Display recent transactions
    st.subheader("Recent Transactions")
    st.dataframe(transactions)

    # Display flagged transactions
    st.subheader("🚩 Flagged Suspicious Transactions")
    st.dataframe(flagged_transactions)

    # Real-time statistics
    st.subheader("Statistics")
    st.metric(label="Total Transactions Processed", value=len(st.session_state["transactions"]))
    st.metric(label="Suspicious Transactions Flagged", value=len(st.session_state["flagged"]))

# Visualization
with st.container():
    st.header("📈 Insights")
    st.write("Visualizations of transaction data and flagged anomalies.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Amount Distribution")
        if not st.session_state["transactions"].empty:
            fig, ax = plt.subplots()
            st.session_state["transactions"]['Amount Received'].hist(ax=ax, bins=20)
            ax.set_title("Transaction Amount Distribution")
            ax.set_xlabel("Amount Received")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        else:
            st.info("No transactions available for distribution.")

    with col2:
        st.subheader("Flagged Transactions by Hour")
        if not st.session_state["flagged"].empty:
            flagged_by_hour = st.session_state["flagged"]['Hour'].value_counts().sort_index()
            fig, ax = plt.subplots()
            flagged_by_hour.plot(kind='bar', ax=ax)
            ax.set_title("Flagged Transactions by Hour")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.info("No flagged transactions to display.")

# Auto-refresh
time.sleep(refresh_interval)
st.rerun()
