# model_utils.py

import os
import urllib.request
import joblib
import pandas as pd
import tensorflow as tf


def download_file_from_github_release(url, filename):
    """
    Downloads a file from GitHub Releases and saves it locally.
    """
    if not os.path.exists(filename):
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename} from {url}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
    else:
        print(f"{filename} already exists locally.")
    return filename

def load_model_from_github(url):
    """
    Downloads and loads a TensorFlow Keras model from GitHub Releases.
    """
    filename = url.split('/')[-1]
    download_file_from_github_release(url, filename)
    loaded_model = joblib.load(filename)
    return loaded_model

def load_scaler_from_github(url):
    """
    Downloads and loads a scaler saved with joblib from GitHub Releases.
    """
    filename = url.split('/')[-1]
    download_file_from_github_release(url, filename)
    scaler = joblib.load(filename)
    return scaler

def load_training_columns_from_github(url):
    """
    Downloads and loads the training columns from GitHub Releases.
    """
    filename = url.split('/')[-1]
    download_file_from_github_release(url, filename)
    training_columns = joblib.load(filename)
    return training_columns

def load_dataset_from_github(url):
    """
    Downloads and loads a dataset from GitHub.
    """
    filename = url.split('/')[-1]
    download_file_from_github_release(url, filename)
    df = pd.read_csv(filename)
    return df
