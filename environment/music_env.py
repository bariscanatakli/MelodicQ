import numpy as np
import pandas as pd

class MusicRecommendationEnv:
    """
    Simulated environment for music recommendations
    """
    def __init__(self, songs_data, feature_cols, num_songs_per_state=5, max_steps=20):
        """
        Initialize the environment
        
        Args:
            songs_data (DataFrame): Processed songs data
            feature_cols (list): List of feature column names
            num_songs_per_state (int): Number of songs to include in each state
            max_steps (int): Maximum number of steps in an episode
        """
        self.songs_data = songs_data
        self.feature_cols = feature_cols
        self.num_songs = len(songs_data)
        self.num_songs_per_state = num_songs_per_state
        self.max_steps = max_steps
        self.feature_dim = len(feature_cols)
        
        # Action space: recommend a song from the catalog
        self.action_space = self.num_songs
        
        # State: features of previously recommended songs + user profile
        self.state_dim = self.feature_dim * self.num_songs_per_state + self.feature_dim
        
        # Initialize user preferences (simulated)
        self.reset()
        
    def reset(self):
        """
        Reset the environment to start a new episode
        
        Returns:
            numpy.array: Initial state
        """
        # Reset step counter
        self.current_step = 0
        
        # Simulate a user with random preferences
        self.user_preferences = np.random.rand(self.feature_dim)
        
        # Initialize history with random songs
        random_indices = np.random.choice(self.num_songs, self.num_songs_per_state)
        self.history = self.songs_data.iloc[random_indices]
        
        # Create initial state
        state = self._get_state()
        
        return state
    
    def step(self, action):
        """
        Take a step in the environment by recommending a song
        
        Args:
            action (int): Index of the song to recommend
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        self.current_step += 1
        
        # Get the recommended song
        if action < 0 or action >= self.num_songs:
            # Invalid action, penalize
            reward = -1.0
            next_state = self._get_state()
            done = self.current_step >= self.max_steps
            info = {'error': 'Invalid action'}
            return next_state, reward, done, info
        
        recommended_song = self.songs_data.iloc[action]
        
        # Calculate reward based on similarity to user preferences
        song_features = recommended_song[self.feature_cols].values
        similarity = 1.0 - np.mean(np.abs(song_features - self.user_preferences))
        
        # Add some noise to make it more realistic
        reward = similarity + np.random.normal(0, 0.1)
        
        # Update history (add new song, remove oldest)
        self.history = pd.concat([self.history.iloc[1:], pd.DataFrame([recommended_song])])
        
        # Update user preferences slightly based on the song
        # This simulates how a user's taste might evolve
        alpha = 0.1  # Rate of preference change
        self.user_preferences = (1 - alpha) * self.user_preferences + alpha * song_features
        
        # Get next state
        next_state = self._get_state()
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        info = {
            'song_id': recommended_song['id'],
            'song_name': recommended_song['name'],
            'similarity': similarity
        }
        
        return next_state, reward, done, info
    
    def _get_state(self):
        """
        Create the state representation
        
        Returns:
            numpy.array: State representation
        """
        # Extract features from history
        history_features = self.history[self.feature_cols].values.flatten()
        
        # Combine with user preferences
        state = np.concatenate([history_features, self.user_preferences])
        
        # Ensure the state is a flat array of float32 type
        return state.astype(np.float32)
