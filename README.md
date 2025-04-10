# MelodicQ

![MelodicQ Logo](./assets/melodicq_logo.png)

A music recommendation system powered by Deep Q-Learning, using reinforcement learning to discover personalized music preferences.

## Overview

MelodicQ takes a novel approach to music recommendations by framing the recommendation problem as a reinforcement learning task. Instead of traditional collaborative filtering or content-based methods, MelodicQ learns to recommend songs by exploring a vast music space and receiving feedback on its recommendations.

### Key Features

- **Reinforcement Learning** approach to music recommendations using Double DQN
- **Personalized recommendations** based on audio features and user preferences
- **Exploration vs. exploitation** balance with epsilon-greedy strategy
- **Comprehensive evaluation metrics** for recommendation quality
- **Visualization tools** for understanding recommendation patterns

## Architecture

MelodicQ consists of several key components:

1. **Environment**: A music environment that presents songs and receives feedback
2. **Agent**: A Deep Q-Network (DQN) that learns to select songs based on user state
3. **Experience Replay**: Memory buffer that stores and replays past experiences
4. **Reward System**: Rewards good recommendations based on user satisfaction

## Installation

```bash
# Clone the repository
git clone https://github.com/bariscanatakli/MelodicQ.git
cd MelodicQ

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python main.py --train --data_path data/tracks_features.csv
```

### Evaluation

```bash
python main.py --evaluate --model_path models/dqn_model_final.pth --data_path data/tracks_features.csv
```

### Visualization

```bash
python main.py --visualize --data_path data/tracks_features.csv
```

## Dataset

MelodicQ works with music datasets containing audio features. We recommend using Spotify's audio features API or pre-extracted datasets containing features like:

- Danceability
- Energy
- Valence
- Tempo
- Acousticness
- Instrumentalness
- Liveness

## Performance

The DQN agent learns to make better recommendations over time, as shown by increasing reward trends during training. Evaluation metrics include:

- Average reward per episode
- Recommendation diversity
- User satisfaction scores
- Song feature distribution

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspiration from advances in Deep Reinforcement Learning
- Built with PyTorch and pandas

## Author

- [Baris Can Atakli](https://github.com/bariscanatakli)
