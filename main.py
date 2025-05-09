import os
import pandas as pd
import torch
import argparse
from tqdm import tqdm

from data.data_loader import load_data
from data.preprocess import preprocess_data
from model.rainbow_agent import RainbowAgent, DEFAULT_CFG     # DQNAgent yerine!
from environment.music_env import MusicRecommendationEnv
from training.train import train_dqn
from evaluation.evaluate import evaluate_agent, visualize_evaluation
from utils.utils import set_seeds, plot_song_features, analyze_recommendations, get_optimal_batch_size
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
    
    # Calculate optimal batch size for GPU if requested
    
    batch_size = Config.BATCH_SIZE
    if args.optimize_batch_size and torch.cuda.is_available():
        batch_size = get_optimal_batch_size(env.state_dim)

    cfg = DEFAULT_CFG.copy()
    cfg["batch_size"] = batch_size
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["buffer_size"] = args.memory_size

    # ---- Agent ----
    agent = RainbowAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space,
        cfg=cfg
    )

    print(f"Agent created (Rainbow) running on {cfg['device']}")
    print(f"Batch size: {agent.batch_size}")
    
    # Print agent details
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Agent created with DQN running on {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {agent.batch_size}")
    # print(f"Epsilon: {agent.epsilon} (min: {agent.epsilon_min}, decay: {agent.epsilon_decay})")
    
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
            log_dir=Config.LOG_DIR,
            enable_gpu_monitoring=not args.disable_gpu_monitoring
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
    parser.add_argument("--optimize_batch_size", action="store_true",
                        help="Optimize batch size based on GPU memory")
    parser.add_argument("--disable_gpu_monitoring", action="store_true",
                        help="Disable GPU monitoring during training")
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes for training')
    parser.add_argument('--sample_size', type=int, default=100000, help='Number of songs to sample (smaller is faster)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--memory_size', type=int, default=20000, help='Replay memory size')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    
    args = parser.parse_args()
    
    # Default to both train and evaluate if neither is specified
    if not args.train and not args.evaluate:
        args.train = True
        args.evaluate = True
        
    main(args)