import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt

# Load the model and scaler
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('https://github.com/tadiwamark/AML/releases/download/dnn_aml/dnn_aml_model.h5') 
    return model

@st.cache_resource
def load_scaler():
    scaler = StandardScaler()
    return scaler

model = load_model()
scaler = load_scaler()

# Preprocessing function
def preprocess_data(data, scaler):
    # Convert 'Timestamp' to datetime and extract features
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Year'] = data['Timestamp'].dt.year
    data['Month'] = data['Timestamp'].dt.month
    data['Day'] = data['Timestamp'].dt.day
    data['Hour'] = data['Timestamp'].dt.hour
    data['Minute'] = data['Timestamp'].dt.minute
    data = data.drop(columns=['Timestamp'])

    # Label encode Account and Account.1 columns (as in training)
    data['Account'] = data['Account'].astype(str).apply(lambda x: hash(x) % (10 ** 6))
    data['Account.1'] = data['Account.1'].astype(str).apply(lambda x: hash(x) % (10 ** 6))

    # Scale numerical features
    numeric_columns = ['Amount Received', 'Amount Paid']
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data

# Generate real-time transactions
def generate_transactions(num_transactions):
    np.random.seed(42)
    return pd.DataFrame({
        'Timestamp': pd.date_range(start='2022-01-01', periods=num_transactions, freq='T'),
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

# Streamlit UI
st.set_page_config(page_title="Real-Time AML Transaction Monitor", layout="wide")

st.title("💸 Real-Time Anti-Money Laundering (AML) Monitor")
st.sidebar.title("Settings")
st.sidebar.markdown("Configure transaction simulation settings.")

# Settings
batch_size = st.sidebar.slider("Batch Size", min_value=10, max_value=100, value=60, step=10)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", min_value=1, max_value=10, value=5, step=1)

# Data storage
if "transactions" not in st.session_state:
    st.session_state["transactions"] = pd.DataFrame()
if "flagged" not in st.session_state:
    st.session_state["flagged"] = pd.DataFrame()

# Main simulation
with st.container():
    st.header("📊 Real-Time Transactions")
    st.write("Simulating real-time transactions and processing them to detect suspicious activity.")

    transactions = generate_transactions(batch_size)
    preprocessed_data = preprocess_data(transactions, scaler)

    # Run model predictions
    predictions = (model.predict(preprocessed_data) > 0.5).astype(int)
    transactions['Is Laundering'] = predictions

    # Append flagged transactions
    flagged_transactions = transactions[transactions['Is Laundering'] == 1]
    st.session_state["transactions"] = pd.concat([st.session_state["transactions"], transactions])
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
    st.write("Visualizations of the transaction data and flagged anomalies.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Amount Distribution")
        fig, ax = plt.subplots()
        st.session_state["transactions"]['Amount Received'].hist(ax=ax, bins=20)
        ax.set_title("Transaction Amount Distribution")
        ax.set_xlabel("Amount Received")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with col2:
        st.subheader("Flagged Transactions by Hour")
        flagged_by_hour = st.session_state["flagged"]['Hour'].value_counts().sort_index()
        fig, ax = plt.subplots()
        flagged_by_hour.plot(kind='bar', ax=ax)
        ax.set_title("Flagged Transactions by Hour")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Count")
        st.pyplot(fig)

# Auto-refresh
time.sleep(refresh_interval)
st.experimental_rerun()
