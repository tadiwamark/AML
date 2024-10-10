import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import time

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define the path to the zip file
zip_file_path = '/content/drive/MyDrive/datasets/dataset.zip'  # Adjust the path accordingly

# Extract the zip file
import zipfile

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('/content/dataset_folder')

# Load the trained model, scaler, and training columns from Google Drive
model_path = '/content/drive/MyDrive/models/gbc_aml_model.pkl'
scaler_path = '/content/drive/MyDrive/models/scaler.save'
columns_path = '/content/drive/MyDrive/models/training_columns.pkl'

gbc = joblib.load(model_path)
scaler = joblib.load(scaler_path)
training_columns = joblib.load(columns_path)

# Load the dataset
dataset_path = '/content/dataset_folder/HI-Small_Trans.csv'
df = pd.read_csv(dataset_path)
df = df.drop_duplicates()
df = df.dropna()

# Preprocess the data
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
