
import numpy as np
import pandas as pd

class MusicRecommendationEnv:
    """
    Simulated environment for music recommendations with improved reward logic.
    """
    def __init__(self, songs_data, feature_cols, num_songs_per_state=5, max_steps=20):
        self.songs_data = songs_data.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.num_songs = len(songs_data)
        self.num_songs_per_state = num_songs_per_state
        self.max_steps = max_steps
        self.feature_dim = len(feature_cols)
        self.action_space = self.num_songs
        self.state_dim = self.feature_dim * self.num_songs_per_state + self.feature_dim
        self.reset()

    def reset(self):
        self.current_step = 0
        self.user_preferences = np.random.rand(self.feature_dim)
        random_indices = np.random.choice(self.num_songs, self.num_songs_per_state, replace=False)
        self.history = self.songs_data.iloc[random_indices]
        return self._get_state()

    def step(self, action):
        self.current_step += 1

        if action < 0 or action >= self.num_songs:
            return self._get_state(), -1.0, self.current_step >= self.max_steps, {'error': 'Invalid action'}

        recommended_song = self.songs_data.iloc[action]
        song_features = recommended_song[self.feature_cols].values

        # Cosine similarity as a more robust similarity metric
        similarity = self._cosine_similarity(song_features, self.user_preferences)

        # New reward shaping: encourage diversity + similarity
        diversity_bonus = self._diversity_bonus(action)
        reward = similarity + 0.1 * diversity_bonus + np.random.normal(0, 0.05)

        self.history = pd.concat([self.history.iloc[1:], pd.DataFrame([recommended_song])], ignore_index=True)

        alpha = 0.1
        self.user_preferences = (1 - alpha) * self.user_preferences + alpha * song_features

        next_state = self._get_state()
        done = self.current_step >= self.max_steps

        info = {
            'song_id': recommended_song.get('id', str(action)),
            'song_name': recommended_song.get('name', 'unknown'),
            'similarity': similarity
        }

        return next_state, reward, done, info

    def _get_state(self):
        history_features = self.history[self.feature_cols].values.flatten()
        state = np.concatenate([history_features, self.user_preferences])
        return state.astype(np.float32)

    def _cosine_similarity(self, vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def _diversity_bonus(self, action):
        if len(self.history) == 0:
            return 1.0
        last_song_ids = set(self.history.get('id', []))
        return 1.0 if action not in last_song_ids else 0.0
