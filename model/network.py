import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    """
    Neural network for the DQN agent
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize the DQN network
        
        Args:
            state_dim (int): Dimension of state
            action_dim (int): Dimension of action
            hidden_dim (int): Dimension of hidden layer
        """
        super(DQNNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (Tensor): Input state
            
        Returns:
            Tensor: Q-values for each action
        """
        return self.layers(x)
