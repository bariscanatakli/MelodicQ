import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import psutil
from datetime import datetime
import os

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
    from collections import Counter

    # Sadece id ve isim alınır (ilk 2 eleman)
    all_recommendations = [s[:2] for s in recommended_songs]

    # Her şarkıdan kaçar defa önerildiğini say
    counter = Counter(all_recommendations)
    most_common = counter.most_common(10)

    unique_songs = len(counter)
    total = len(all_recommendations)
    diversity_ratio = unique_songs / total if total > 0 else 0.0

    result = {
        "total_recommendations": total,
        "unique_songs": unique_songs,
        "diversity_ratio": diversity_ratio,
        "most_recommended": [
            (sid, name, songs_data[songs_data["id"] == sid]["artists"].values[0], count)
            for ((sid, name), count) in most_common
        ]
    }

    return result


def plot_rewards(rewards, filepath):
    """
    Plot the rewards over episodes
    
    Args:
        rewards (list): List of rewards
        filepath (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode')
    plt.savefig(filepath)
    plt.close()

def monitor_gpu_usage():
    """
    Monitor GPU memory usage if CUDA is available
    
    Returns:
        dict: GPU memory statistics or None if CUDA is not available
    """
    if not torch.cuda.is_available():
        return None
    
    try:
        # Get GPU memory stats
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = t - (r + a)  # free memory
        
        return {
            'total': t / (1024**3),  # Convert to GB
            'reserved': r / (1024**3),
            'allocated': a / (1024**3),
            'free': f / (1024**3)
        }
    except Exception as e:
        print(f"Error monitoring GPU: {e}")
        return None

def log_gpu_usage(episode, batch_size, logs_dir="logs"):
    """
    Log GPU memory usage to a text file
    
    Args:
        episode (int): Current episode
        batch_size (int): Current batch size
        logs_dir (str): Directory to save logs
    """
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        
    gpu_stats = monitor_gpu_usage()
    if gpu_stats is None:
        return
    
    log_file = os.path.join(logs_dir, "gpu_usage.txt")
    with open(log_file, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] Episode: {episode}, Batch: {batch_size}, "
        log_line += f"GPU Memory (GB): Total={gpu_stats['total']:.2f}, "
        log_line += f"Used={gpu_stats['allocated']:.2f}, "
        log_line += f"Reserved={gpu_stats['reserved']:.2f}, "
        log_line += f"Free={gpu_stats['free']:.2f}\n"
        f.write(log_line)

def get_optimal_batch_size(input_dim):
    """
    Calculate an optimal batch size based on GPU memory
    
    Args:
        input_dim (int): Dimension of input data
        
    Returns:
        int: Suggested optimal batch size
    """
    if not torch.cuda.is_available():
        return 64  # Default batch size for CPU
    
    gpu_stats = monitor_gpu_usage()
    if gpu_stats is None:
        return 64
    
    # Estimate bytes per sample (rough estimation)
    bytes_per_sample = input_dim * 4  # 4 bytes per float32
    
    # Leave 20% of GPU memory free for other operations
    usable_memory = gpu_stats['total'] * 0.8 * (1024**3)  # Convert to bytes
    
    # Calculate optimal batch size
    optimal_batch = int(usable_memory / (bytes_per_sample * 10))  # Factor of 10 for safety
    
    # Limit to reasonable range (power of 2 is often efficient)
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    for size in batch_sizes:
        if size >= optimal_batch:
            return size
    
    return 64  # Default if something goes wrong
