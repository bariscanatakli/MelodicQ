import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    """
    Preprocess the Spotify dataset for use in the DQN model
    
    Args:
        data (DataFrame): Raw dataset
    
    Returns:
        DataFrame: Processed dataset with selected features
    """
    # Select the key features for the initial phase
    key_features = ['danceability', 'energy', 'tempo', 'valence']
    
    # Check if all features exist in the dataset
    missing_features = [f for f in key_features if f not in data.columns]
    if missing_features:
        raise ValueError(f"Dataset is missing these features: {missing_features}")
    
    # Select relevant columns (including ID and key features)
    processed_data = data[['id', 'name', 'artists'] + key_features].copy()
    
    # Normalize features to [0, 1] range
    scaler = MinMaxScaler()
    processed_data[key_features] = scaler.fit_transform(processed_data[key_features])
    
    # Remove rows with missing values
    processed_data = processed_data.dropna()
    
    print(f"Preprocessed data shape: {processed_data.shape}")
    return processed_data
