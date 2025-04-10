import os
import pandas as pd
import torch
import argparse
from tqdm import tqdm

from data.data_loader import load_data
from data.preprocess import preprocess_data
from model.dqn_agent import DQNAgent
from environment.music_env import MusicRecommendationEnv
from training.train import train_dqn
from evaluation.evaluate import evaluate_agent, visualize_evaluation
from utils.utils import set_seeds, plot_song_features, analyze_recommendations
from config import Config

def main(args):
    # Set random seeds
    set_seeds(Config.RANDOM_SEED)
    
    # Create directories
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # Load and preprocess data
    data = load_data(args.data_path, sample_size=Config.SAMPLE_SIZE, random_state=Config.RANDOM_SEED)
    processed_data = preprocess_data(data)
    
    # Visualize sample data if requested
    if args.visualize:
        plot_song_features(processed_data, Config.FEATURE_COLS, n_songs=10)
    
    # Create environment
    env = MusicRecommendationEnv(
        processed_data,
        Config.FEATURE_COLS,
        num_songs_per_state=Config.NUM_SONGS_PER_STATE,
        max_steps=Config.MAX_STEPS
    )
    
    # Print environment details
    print(f"Environment created with {len(processed_data)} songs")
    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_space}")
    
    # Create agent
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space,
        hidden_dim=Config.HIDDEN_DIM,
        lr=Config.LEARNING_RATE,
        gamma=Config.GAMMA,
        epsilon=Config.EPSILON,
        epsilon_min=Config.EPSILON_MIN,
        epsilon_decay=Config.EPSILON_DECAY,
        memory_size=Config.MEMORY_SIZE,
        batch_size=Config.BATCH_SIZE
    )
    
    # Print agent details
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Agent created with DQN running on {device}")
    print(f"Epsilon: {agent.epsilon} (min: {agent.epsilon_min}, decay: {agent.epsilon_decay})")
    
    # Training
    if args.train:
        print("Starting training...")
        episode_rewards, avg_rewards = train_dqn(
            agent=agent,
            env=env,
            num_episodes=Config.NUM_EPISODES,
            target_update_freq=Config.TARGET_UPDATE_FREQ,
            eval_freq=Config.EVAL_FREQ,
            save_dir=Config.MODEL_SAVE_DIR,
            log_dir=Config.LOG_DIR
        )
        print("Training completed!")
    
    # Load model for evaluation
    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        agent.load(args.model_path)
    elif args.train:
        # Use the final model from training
        model_path = os.path.join(Config.MODEL_SAVE_DIR, 'dqn_model_final.pth')
        print(f"Using final trained model: {model_path}")
        agent.load(model_path)
    
    # Evaluation
    if args.evaluate:
        print("Starting evaluation...")
        metrics = evaluate_agent(
            agent=agent,
            env=env,
            num_episodes=Config.EVAL_EPISODES
        )
        
        # Visualize evaluation results
        visualize_evaluation(
            metrics=metrics,
            save_path=os.path.join(Config.LOG_DIR, 'evaluation_results.png')
        )
        
        # Analyze recommendations
        analysis = analyze_recommendations(
            recommended_songs=metrics['recommended_songs'],
            songs_data=processed_data
        )
        
        print("\nRecommendation Analysis:")
        print(f"Total recommendations: {analysis['total_recommendations']}")
        print(f"Unique songs recommended: {analysis['unique_songs']}")
        print(f"Diversity ratio: {analysis['diversity_ratio']:.4f}")
        
        print("\nMost recommended songs:")
        for song_id, name, artists, count in analysis['most_recommended']:
            print(f"{name} by {artists}: recommended {count} times")
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music Recommendation System with DQN")
    
    parser.add_argument("--data_path", type=str, default=Config.DATA_PATH,
                        help="Path to the dataset file")
    parser.add_argument("--train", action="store_true",
                        help="Train the agent")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the agent")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a saved model for evaluation")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize sample data")
    
    args = parser.parse_args()
    
    # Default to both train and evaluate if neither is specified
    if not args.train and not args.evaluate:
        args.train = True
        args.evaluate = True
        
    main(args)