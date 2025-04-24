
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_agent(agent, env, num_episodes=100):
    """
    Evaluate the trained Rainbow DQN agent

    Args:
        agent: Trained Rainbow DQN agent
        env: Music recommendation environment
        num_episodes (int): Number of episodes for evaluation

    Returns:
        dict: Evaluation metrics
    """
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
            action = agent.select_action(state, mode='eval')
            next_state, reward, done, info = env.step(action)
            episode_similarities.append(info.get('similarity', 0))
            episode_songs.append((info.get('song_id', ''), info.get('song_name', '')))
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        song_similarities.append(np.mean(episode_similarities))
        recommended_songs.extend(episode_songs)

    unique_songs = set(s[0] for s in recommended_songs)
    diversity_ratio = len(unique_songs) / len(recommended_songs)

    song_counts = {}
    for song_id, name in recommended_songs:
        key = (song_id, name)
        song_counts[key] = song_counts.get(key, 0) + 1

    most_recommended = sorted([(sid, name, count) for (sid, name), count in song_counts.items()],
                              key=lambda x: x[2], reverse=True)[:10]

    metrics = {
        'avg_reward': np.mean(episode_rewards),
        'avg_similarity': np.mean(song_similarities),
        'episode_rewards': episode_rewards,
        'song_similarities': song_similarities,
        'recommended_songs': recommended_songs,
        'diversity_ratio': diversity_ratio,
        'most_recommended': most_recommended,
        'total_recommendations': len(recommended_songs),
        'unique_recommendations': len(unique_songs)
    }

    return metrics


def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def visualize_evaluation(metrics, save_path=None):
    rewards = metrics['episode_rewards']
    similarities = metrics['song_similarities']

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    axs[0, 0].hist(rewards, bins=20, alpha=0.7, color='skyblue')
    axs[0, 0].axvline(np.mean(rewards), color='red', linestyle='--', label='Avg Reward')
    axs[0, 0].set_title(f'Reward Distribution\n(Avg: {np.mean(rewards):.2f})')
    axs[0, 0].legend()

    axs[0, 1].hist(similarities, bins=20, alpha=0.7, color='orchid')
    axs[0, 1].axvline(np.mean(similarities), color='red', linestyle='--', label='Avg Similarity')
    axs[0, 1].set_title(f'Similarity Distribution\n(Avg: {np.mean(similarities):.2f})')
    axs[0, 1].legend()

    axs[0, 2].plot(rewards, label='Reward')
    axs[0, 2].plot(moving_average(rewards), label='Moving Avg', linewidth=2)
    axs[0, 2].set_title('Reward per Episode')
    axs[0, 2].legend()

    axs[1, 0].plot(similarities, label='Similarity', color='darkorange')
    axs[1, 0].plot(moving_average(similarities), label='Moving Avg', linewidth=2)
    axs[1, 0].set_title('Similarity per Episode')
    axs[1, 0].legend()

    axs[1, 1].plot(np.cumsum(rewards), label='Cumulative Reward', color='seagreen')
    axs[1, 1].set_title('Cumulative Reward')
    axs[1, 1].legend()

    axs[1, 2].scatter(similarities, rewards, alpha=0.6, color='mediumslateblue')
    axs[1, 2].set_title('Reward vs. Similarity')
    axs[1, 2].set_xlabel('Similarity')
    axs[1, 2].set_ylabel('Reward')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
