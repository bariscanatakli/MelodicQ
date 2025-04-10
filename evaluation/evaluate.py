import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_agent(agent, env, num_episodes=100):
    """
    Evaluate the trained DQN agent
    
    Args:
        agent: Trained DQN agent
        env: Music recommendation environment
        num_episodes (int): Number of episodes for evaluation
        
    Returns:
        dict: Evaluation metrics
    """
    # Save original epsilon and set to minimum for evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = agent.epsilon_min
    
    episode_rewards = []
    song_similarities = []
    recommended_songs = []
    
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        total_reward = 0
        episode_similarities = []
        episode_songs = []
        
        done = False
        while not done:
            # Select action (using greedy policy for evaluation)
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Log data
            episode_similarities.append(info.get('similarity', 0))
            episode_songs.append((info.get('song_id', ''), info.get('song_name', '')))
            
            # Update state and total reward
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        song_similarities.append(np.mean(episode_similarities))
        recommended_songs.append(episode_songs)
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    # Calculate metrics
    avg_reward = np.mean(episode_rewards)
    avg_similarity = np.mean(song_similarities)
    
    metrics = {
        'avg_reward': avg_reward,
        'avg_similarity': avg_similarity,
        'episode_rewards': episode_rewards,
        'song_similarities': song_similarities,
        'recommended_songs': recommended_songs
    }
    
    return metrics

def visualize_evaluation(metrics, save_path=None):
    """
    Visualize evaluation results
    
    Args:
        metrics (dict): Evaluation metrics
        save_path (str): Path to save the visualization
    """
    plt.figure(figsize=(15, 10))
    
    # Plot rewards histogram
    plt.subplot(2, 2, 1)
    plt.hist(metrics['episode_rewards'], bins=20, alpha=0.7)
    plt.axvline(metrics['avg_reward'], color='r', linestyle='dashed', linewidth=2)
    plt.title(f'Reward Distribution (Avg: {metrics["avg_reward"]:.4f})')
    plt.xlabel('Reward')
    plt.ylabel('Count')
    
    # Plot similarities histogram
    plt.subplot(2, 2, 2)
    plt.hist(metrics['song_similarities'], bins=20, alpha=0.7)
    plt.axvline(metrics['avg_similarity'], color='r', linestyle='dashed', linewidth=2)
    plt.title(f'Song Similarity Distribution (Avg: {metrics["avg_similarity"]:.4f})')
    plt.xlabel('Similarity')
    plt.ylabel('Count')
    
    # Plot rewards over episodes
    plt.subplot(2, 2, 3)
    plt.plot(metrics['episode_rewards'])
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot similarities over episodes
    plt.subplot(2, 2, 4)
    plt.plot(metrics['song_similarities'])
    plt.title('Average Song Similarity per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Similarity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
