import os

class Config:
    # Paths
    DATA_PATH = "data/tracks_features.csv"
    MODEL_SAVE_DIR = "saved_models"
    LOG_DIR = "logs"
    
    # Data preprocessing
    SAMPLE_SIZE = 100000  # Number of songs to sample from dataset
    RANDOM_SEED = 42
    
    # Feature columns to use
    FEATURE_COLS = ['danceability', 'energy', 'tempo', 'valence']
    
    # Environment settings
    NUM_SONGS_PER_STATE = 5  # Number of songs in history to include in state
    MAX_STEPS = 20  # Maximum number of steps per episode
    
    # Agent settings
    HIDDEN_DIM = 128
    LEARNING_RATE = 1e-4
    GAMMA = 0.99  # Discount factor
    EPSILON = 1.0  # Initial exploration rate
    EPSILON_MIN = 0.01  # Minimum exploration rate
    EPSILON_DECAY = 0.995  # Decay rate for exploration
    MEMORY_SIZE = 100000  # Size of replay memory
    BATCH_SIZE = 64  # Batch size for training
    
    # Training settings
    NUM_EPISODES = 1000
    TARGET_UPDATE_FREQ = 10  # Frequency of target network updates
    EVAL_FREQ = 100  # Frequency of evaluation during training
    
    # Evaluation settings
    EVAL_EPISODES = 100  # Number of episodes for final evaluation
