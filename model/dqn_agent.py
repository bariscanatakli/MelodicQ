import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from model.network import DQNNetwork

class DQNAgent:
    """
    Deep Q-Network agent for music recommendation
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-4, 
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, 
                 epsilon_decay=0.995, memory_size=10000, batch_size=64,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the DQN agent
        
        Args:
            state_dim (int): Dimension of state
            action_dim (int): Dimension of action
            hidden_dim (int): Dimension of hidden layer
            lr (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Initial exploration rate
            epsilon_min (float): Minimum exploration rate
            epsilon_decay (float): Decay rate for exploration
            memory_size (int): Size of replay memory
            batch_size (int): Batch size for training
            device (str): Device to use for training
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device
        
        # Q-Network and target network
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Replay memory
        self.memory = deque(maxlen=memory_size)
        
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        """
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            state (numpy.array): Current state
            
        Returns:
            int: Selected action
        """
        if np.random.rand() <= self.epsilon:
            # Exploration: random action
            return random.randrange(self.action_dim)
        
        # Exploitation: best action from Q-network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()
    
    def replay(self):
        """
        Train the model using experience replay
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of experiences
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = []
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            
            # Get current Q values
            current_q = self.q_network(state_tensor)
            
            # Get target Q values
            target_q = current_q.clone().detach()
            
            if done:
                target_q[action] = reward
            else:
                with torch.no_grad():
                    # Double DQN: use online network to select action, target network to evaluate
                    next_action = torch.argmax(self.q_network(next_state_tensor)).item()
                    max_next_q = self.target_network(next_state_tensor)[next_action].item()
                target_q[action] = reward + self.gamma * max_next_q
            
            states.append(state_tensor)
            targets.append(target_q)
        
        # Train the network
        states = torch.stack(states)
        targets = torch.stack(targets)
        
        self.optimizer.zero_grad()
        predictions = self.q_network(states)
        loss = self.loss_fn(predictions, targets)
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_network(self):
        """
        Update the target network with the Q-network's weights
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def save(self, filepath):
        """
        Save the model
        """
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        
    def load(self, filepath):
        """
        Load the model
        """
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
