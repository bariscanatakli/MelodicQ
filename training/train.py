import numpy as np
import torch
import os
import time
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import plot_rewards, monitor_gpu_usage, log_gpu_usage, get_optimal_batch_size
from model.rainbow_agent import RainbowAgent, DEFAULT_CFG

def train_dqn(agent, env, num_episodes=1000, target_update_freq=10, eval_freq=100, 
              save_dir='saved_models', log_dir='logs', enable_gpu_monitoring=True,
              early_stopping=False, patience=50, min_delta=0.01):
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
        enable_gpu_monitoring (bool): Whether to monitor GPU usage
        early_stopping (bool): Whether to use early stopping
        patience (int): Number of episodes to wait for improvement
        min_delta (float): Minimum change to qualify as improvement
    """
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize lists to track progress
    episode_rewards = []
    avg_rewards = []
    
    start_time = time.time()
    
    # Log GPU info at the start
    if enable_gpu_monitoring and torch.cuda.is_available():
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        print(f"Batch size: {agent.batch_size}")
        # Log initial GPU status
        log_gpu_usage(0, agent.batch_size, log_dir)
    else:
        print("Training on CPU")
    
    # Early stopping variables
    best_avg_reward = float('-inf')
    no_improvement_count = 0
    
    # Training loop
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        max_steps_per_episode = 1000  # Safety limit to prevent infinite loops
        
        while not done and step_count < max_steps_per_episode:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update state and total reward
            state = next_state
            total_reward += reward
            
            # Train the agent (only every few steps for very large action spaces)
            if len(agent.buffer) >= agent.batch_size:
                if agent.action_dim <= 10000 or step_count % 5 == 0:  # Reduce frequency for large action spaces
                    result = agent.train_step()
                    if result:
                        print(f"Loss: {result['loss']:.4f}, TD Error: {result['td_error']:.4f}")
            
            step_count += 1
            
            # Prevent infinite loops
            if step_count >= max_steps_per_episode:
                print(f"Warning: Episode {episode} reached the maximum step limit.")
                done = True
        
        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.soft_update()
        
        # Save episode reward
        episode_rewards.append(total_reward)
        current_avg_reward = np.mean(episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards)
        avg_rewards.append(current_avg_reward)
        
        # Monitor GPU usage periodically
        if enable_gpu_monitoring and episode % 10 == 0 and torch.cuda.is_available():
            log_gpu_usage(episode, agent.batch_size, log_dir)
        
        # Early stopping check
        if early_stopping:
            if current_avg_reward > best_avg_reward + min_delta:
                best_avg_reward = current_avg_reward
                no_improvement_count = 0
                # Save the best model
                best_model_path = os.path.join(save_dir, 'dqn_model_best.pth')
                agent.save(best_model_path)
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= patience:
                print(f"\nEarly stopping triggered after {episode+1} episodes")
                print(f"Best average reward: {best_avg_reward:.4f}")
                # Load the best model
                best_model_path = os.path.join(save_dir, 'dqn_model_best.pth')
                if os.path.exists(best_model_path):
                    agent.load(best_model_path)
                break
        
        # Evaluate and save model periodically
        if episode % eval_freq == 0:
            # Save model
            model_path = os.path.join(save_dir, f'dqn_model_episode_{episode}.pth')
            agent.save(model_path)
            
            # Log progress
            elapsed_time = time.time() - start_time
            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"Avg Reward (last 100): {avg_rewards[-1]:.4f}")
            print(f"Time Elapsed: {elapsed_time:.2f} seconds")
            
            # Log GPU status during evaluation
            if enable_gpu_monitoring and torch.cuda.is_available():
                gpu_stats = monitor_gpu_usage()
                if gpu_stats:
                    print(f"GPU Memory: {gpu_stats['allocated']:.2f}GB used / {gpu_stats['total']:.2f}GB total")
            
            # Plot rewards
            plot_path = os.path.join(log_dir, f'rewards_episode_{episode}.png')
            plot_rewards(episode_rewards, plot_path)
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'dqn_model_final.pth')
    agent.save(final_model_path)
    
    # Save rewards history
    rewards_df = pd.DataFrame({
        'episode': range(len(episode_rewards)),
        'reward': episode_rewards,
        'avg_reward': avg_rewards
    })
    rewards_df.to_csv(os.path.join(log_dir, 'rewards_history.csv'), index=False)
    
    # Final GPU usage log
    if enable_gpu_monitoring and torch.cuda.is_available():
        log_gpu_usage(num_episodes, agent.batch_size, log_dir)
    
    return episode_rewards, avg_rewards
