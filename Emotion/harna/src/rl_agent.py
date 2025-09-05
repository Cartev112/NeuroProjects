"""
Reinforcement learning agent with intrinsic emotional rewards/costs.

Implements RL where emotions provide additional reward signals,
shaping exploration-exploitation trade-offs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from collections import deque
import random


class EmotionalRewardFunction:
    """
    Computes intrinsic emotional rewards from emotional states.
    
    Positive emotions provide rewards, negative emotions provide costs.
    """
    
    def __init__(
        self,
        valence_weight: float = 1.0,
        arousal_weight: float = 0.3,
        novelty_bonus: float = 0.2
    ):
        """
        Initialize emotional reward function.
        
        Args:
            valence_weight: Weight for valence contribution
            arousal_weight: Weight for arousal contribution
            novelty_bonus: Bonus for novel experiences
        """
        self.valence_weight = valence_weight
        self.arousal_weight = arousal_weight
        self.novelty_bonus = novelty_bonus
    
    def compute_reward(
        self,
        valence: float,
        arousal: float,
        novelty: float = 0.0
    ) -> float:
        """
        Compute emotional reward.
        
        Args:
            valence: Emotional valence (-1 to 1)
            arousal: Arousal level (-1 to 1)
            novelty: Novelty level (0 to 1)
            
        Returns:
            Emotional reward
        """
        # Base reward from valence
        reward = self.valence_weight * valence
        
        # Arousal modulation (high arousal amplifies reward/cost)
        arousal_factor = 1.0 + self.arousal_weight * abs(arousal)
        reward *= arousal_factor
        
        # Novelty bonus (encourages exploration)
        reward += self.novelty_bonus * novelty
        
        return reward


class QNetwork(nn.Module):
    """Q-network for Deep Q-Learning."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [128, 128]
    ):
        """Initialize Q-network."""
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(state)


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for policy gradient methods."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ):
        """Initialize actor-critic network."""
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value function)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of (action_probs, state_value)
        """
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        
        return action_probs, state_value


class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity: int = 10000):
        """Initialize replay buffer."""
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class EmotionalRLAgent:
    """
    RL agent with emotional reward shaping.
    
    Combines environmental rewards with intrinsic emotional rewards
    to guide learning and behavior.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        harna_model=None,
        algorithm: str = 'dqn',
        gamma: float = 0.99,
        lr: float = 0.001,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        emotional_weight: float = 0.5,
        device: str = 'cpu'
    ):
        """
        Initialize emotional RL agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            harna_model: HARNA model for emotional processing
            algorithm: RL algorithm ('dqn' or 'actor_critic')
            gamma: Discount factor
            lr: Learning rate
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
            emotional_weight: Weight for emotional rewards
            device: Computing device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.harna_model = harna_model
        self.algorithm = algorithm
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.emotional_weight = emotional_weight
        self.device = device
        
        # Initialize network
        if algorithm == 'dqn':
            self.q_network = QNetwork(state_dim, action_dim).to(device)
            self.target_network = QNetwork(state_dim, action_dim).to(device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
            
        elif algorithm == 'actor_critic':
            self.ac_network = ActorCriticNetwork(state_dim, action_dim).to(device)
            self.optimizer = torch.optim.Adam(self.ac_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Emotional reward function
        self.emotional_reward_fn = EmotionalRewardFunction()
        
        # Statistics
        self.episode_rewards = []
        self.episode_emotional_rewards = []
    
    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy or policy.
        
        Args:
            state: Current state
            explore: Whether to explore
            
        Returns:
            Selected action
        """
        if self.algorithm == 'dqn':
            # Epsilon-greedy
            if explore and np.random.rand() < self.epsilon:
                return np.random.randint(self.action_dim)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        
        elif self.algorithm == 'actor_critic':
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_probs, _ = self.ac_network(state_tensor)
                
                if explore:
                    # Sample from policy
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample()
                else:
                    # Greedy
                    action = action_probs.argmax()
                
                return action.item()
    
    def compute_emotional_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray
    ) -> float:
        """
        Compute emotional reward for transition.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            Emotional reward
        """
        if self.harna_model is None:
            # No emotional processing, return 0
            return 0.0
        
        # Process next state through HARNA to get emotional response
        # (Simplified - in real implementation, would convert state to stimulus)
        features = next_state  # Assume state can be used as features
        
        try:
            response = self.harna_model.process_simple(features)
            
            # Compute emotional reward
            emotional_reward = self.emotional_reward_fn.compute_reward(
                valence=response.get('valence', 0.0),
                arousal=response.get('arousal', 0.0),
                novelty=response.get('novelty', 0.0)
            )
            
            return emotional_reward
        except:
            # Fallback if HARNA processing fails
            return 0.0
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        batch_size: int = 32
    ):
        """
        Update agent based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Environmental reward
            next_state: Next state
            done: Episode done flag
            batch_size: Batch size for updates
        """
        # Compute emotional reward
        emotional_reward = self.compute_emotional_reward(state, action, next_state)
        
        # Combined reward
        total_reward = reward + self.emotional_weight * emotional_reward
        
        # Store in replay buffer
        self.replay_buffer.push(state, action, total_reward, next_state, done)
        
        # Update network
        if len(self.replay_buffer) >= batch_size:
            if self.algorithm == 'dqn':
                self._update_dqn(batch_size)
            elif self.algorithm == 'actor_critic':
                self._update_actor_critic(state, action, total_reward, next_state, done)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _update_dqn(self, batch_size: int):
        """Update DQN."""
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _update_actor_critic(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Update Actor-Critic."""
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        
        # Forward pass
        action_probs, state_value = self.ac_network(state_tensor)
        
        with torch.no_grad():
            _, next_state_value = self.ac_network(next_state_tensor)
            
            if done:
                target_value = reward_tensor
            else:
                target_value = reward_tensor + self.gamma * next_state_value
        
        # Advantage
        advantage = target_value - state_value
        
        # Actor loss (policy gradient)
        log_prob = torch.log(action_probs.squeeze()[action_tensor])
        actor_loss = -log_prob * advantage.detach()
        
        # Critic loss
        critic_loss = F.mse_loss(state_value, target_value)
        
        # Total loss
        loss = actor_loss + critic_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        """Update target network (for DQN)."""
        if self.algorithm == 'dqn':
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath: str):
        """Save agent."""
        if self.algorithm == 'dqn':
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon
            }, filepath)
        elif self.algorithm == 'actor_critic':
            torch.save({
                'ac_network': self.ac_network.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, filepath)
    
    def load(self, filepath: str):
        """Load agent."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if self.algorithm == 'dqn':
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
        elif self.algorithm == 'actor_critic':
            self.ac_network.load_state_dict(checkpoint['ac_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
