import pandas as pd
import numpy as np

def load_data(file_path, sample_size=100000, random_state=42):
    """
    Load a subset of the Spotify dataset
    
    Args:
        file_path (str): Path to the tracks_features.csv file
        sample_size (int): Number of songs to sample
        random_state (int): Random seed for reproducibility
    
    Returns:
        DataFrame: Sampled dataset
    """
    # Load the full dataset
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    
    # Sample a subset if needed
    if sample_size and sample_size < len(data):
        data = data.sample(n=sample_size, random_state=random_state)
        print(f"Sampled {sample_size} songs from the dataset.")
    else:
        print(f"Using full dataset with {len(data)} songs.")
    
    return data
