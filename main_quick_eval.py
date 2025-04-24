import os
import pandas as pd
import torch
import argparse
from tqdm import tqdm

from data.data_loader import load_data
from data.preprocess import preprocess_data
from model.rainbow_agent import RainbowAgent, DEFAULT_CFG
from environment.music_env import MusicRecommendationEnv
from training.train import train_dqn
from evaluation.evaluate import evaluate_agent, visualize_evaluation
from utils.utils import set_seeds, plot_song_features, analyze_recommendations, get_optimal_batch_size
from config import Config

def main(args):
    # Set random seeds
    set_seeds(Config.RANDOM_SEED)

    # Load and preprocess data
    sample_size = 300000
    if args.quick_eval:
        max_steps = 5

    data = load_data(args.data_path, sample_size=sample_size, random_state=Config.RANDOM_SEED)
    processed_data = preprocess_data(data)

    # Create environment
    max_steps = Config.MAX_STEPS
    if args.quick_eval:
        max_steps = 5

    env = MusicRecommendationEnv(
        processed_data,
        Config.FEATURE_COLS,
        num_songs_per_state=Config.NUM_SONGS_PER_STATE,
        max_steps=max_steps
    )

    # Log environment
    print(f"Environment created with {len(processed_data)} songs")
    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_space}")

    # Device and batch
    cfg = DEFAULT_CFG.copy()
    cfg["device"] = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    cfg["batch_size"] = args.batch_size
    cfg["buffer_size"] = args.memory_size

    # Agent
    agent = RainbowAgent(state_dim=env.state_dim, action_dim=env.action_space, cfg=cfg)

    print(f"Agent created (Rainbow) running on {cfg['device']}")
    print(f"Batch size: {agent.batch_size}")

    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        agent.load(args.model_path)

    if args.evaluate:
        num_eval_episodes = 10 if args.quick_eval else Config.EVAL_EPISODES
        print("Starting evaluation...")
        metrics = evaluate_agent(agent=agent, env=env, num_episodes=10)
        visualize_evaluation(metrics=metrics, save_path=os.path.join(Config.LOG_DIR, 'evaluation_results.png'))

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
    parser.add_argument("--data_path", type=str, default=Config.DATA_PATH)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--optimize_batch_size", action="store_true")
    parser.add_argument("--disable_gpu_monitoring", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU instead of GPU")
    parser.add_argument("--quick_eval", action="store_true", help="Run fast evaluation (fewer songs, fewer steps)")
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--sample_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--memory_size', type=int, default=20000)
    parser.add_argument('--epsilon_decay', type=float, default=0.995)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=50)

    args = parser.parse_args()

    if not args.train and not args.evaluate:
        args.train = True
        args.evaluate = True

    main(args)
