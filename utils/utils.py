import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def set_seeds(seed=42):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_song_features(data, feature_cols, n_songs=10):
    """
    Visualize features of a sample of songs
    
    Args:
        data (DataFrame): Songs data
        feature_cols (list): Feature column names
        n_songs (int): Number of songs to visualize
    """
    # Sample n_songs from the dataset
    sample = data.sample(n=n_songs)
    
    # Melt the dataframe for easier plotting
    melted = pd.melt(
        sample[['name'] + feature_cols], 
        id_vars=['name'],
        value_vars=feature_cols,
        var_name='Feature',
        value_name='Value'
    )
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='name', y='Value', hue='Feature', data=melted)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Audio Features for {n_songs} Random Songs')
    plt.tight_layout()
    plt.show()

def analyze_recommendations(recommended_songs, songs_data):
    """
    Analyze the diversity and patterns in recommended songs
    
    Args:
        recommended_songs (list): List of recommended song IDs per episode
        songs_data (DataFrame): Songs data
        
    Returns:
        dict: Analysis results
    """
    # Flatten the list of recommendations
    all_recommendations = [song for episode in recommended_songs for song in episode]
    
    # Count unique songs
    unique_songs = set([song[0] for song in all_recommendations])
    
    # Get most recommended songs
    song_counts = {}
    for song_id, song_name in all_recommendations:
        if song_id in song_counts:
            song_counts[song_id] += 1
        else:
            song_counts[song_id] = 1
    
    # Sort by count
    most_recommended = sorted(
        [(song_id, count) for song_id, count in song_counts.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    # Convert to readable format
    most_recommended_info = []
    for song_id, count in most_recommended:
        song_info = songs_data[songs_data['id'] == song_id]
        if not song_info.empty:
            name = song_info.iloc[0]['name']
            artists = song_info.iloc[0]['artists']
            most_recommended_info.append((song_id, name, artists, count))
    
    return {
        'total_recommendations': len(all_recommendations),
        'unique_songs': len(unique_songs),
        'diversity_ratio': len(unique_songs) / len(all_recommendations),
        'most_recommended': most_recommended_info
    }
