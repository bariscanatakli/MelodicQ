import numpy as np
import torch
import os
import time
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def train_dqn(agent, env, num_episodes=1000, target_update_freq=10, eval_freq=100, 
              save_dir='saved_models', log_dir='logs'):
    """
    Train the DQN agent
    
    Args:
        agent: DQN agent
        env: Music recommendation environment
        num_episodes (int): Number of episodes for training
        target_update_freq (int): Frequency of target network updates
        eval_freq (int): Frequency of evaluation
        save_dir (str): Directory to save models
        log_dir (str): Directory to save logs
    """
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize lists to track progress
    episode_rewards = []
    avg_rewards = []
    
    start_time = time.time()
    
    # Training loop
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and total reward
            state = next_state
            total_reward += reward
            
            # Train the agent
            agent.replay()
        
        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        # Save episode reward
        episode_rewards.append(total_reward)
        avg_rewards.append(np.mean(episode_rewards[-100:]))
        
        # Evaluate and save model periodically
        if episode % eval_freq == 0:
            # Save model
            model_path = os.path.join(save_dir, f'dqn_model_episode_{episode}.pth')
            agent.save(model_path)
            
            # Log progress
            elapsed_time = time.time() - start_time
            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"Avg Reward (last 100): {avg_rewards[-1]:.4f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print(f"Time Elapsed: {elapsed_time:.2f} seconds")
            
            # Plot rewards
            plt.figure(figsize=(10, 5))
            plt.plot(episode_rewards, label='Episode Reward')
            plt.plot(avg_rewards, label='Avg Reward (100 episodes)')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.title('Training Progress')
            plt.savefig(os.path.join(log_dir, f'rewards_episode_{episode}.png'))
            plt.close()
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'dqn_model_final.pth')
    agent.save(final_model_path)
    
    # Save rewards history
    rewards_df = pd.DataFrame({
        'episode': range(num_episodes),
        'reward': episode_rewards,
        'avg_reward': avg_rewards
    })
    rewards_df.to_csv(os.path.join(log_dir, 'rewards_history.csv'), index=False)
    
    return episode_rewards, avg_rewards
