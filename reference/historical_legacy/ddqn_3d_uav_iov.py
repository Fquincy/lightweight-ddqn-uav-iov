"""
3D Double Deep Q-Network (DDQN) Architecture for UAV-IoV Task Offloading
================================================================================
Features:
- 3D Neural Network Architecture for spatial-temporal feature extraction
- Dual Q-Networks (Main and Target) with soft updates
- Prioritized Experience Replay with importance sampling
- Optimized network design with residual connections and attention mechanisms
- Multi-agent support for UAV coordination
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random
from collections import deque
import heapq
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 3D DDQN NEURAL NETWORK ARCHITECTURE
# ============================================================================

class SpatialTemporalEncoder(nn.Module):
    """
    3D Encoder for spatial-temporal feature extraction
    Processes spatial coordinates (x, y, z) and temporal sequences
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(SpatialTemporalEncoder, self).__init__()
        
        # 3D spatial feature extraction
        self.spatial_conv1 = nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.spatial_conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1)
        
        # Temporal feature extraction
        self.temporal_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2)
        
        # Attention mechanism for spatial relationships
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, features)
        Returns:
            Encoded features of shape (batch, hidden_dim)
        """
        batch_size, seq_len, features = x.shape
        
        # Reshape for 1D convolution (treating features as channels)
        x_reshaped = x.transpose(1, 2)  # (batch, features, seq_len)
        
        # Spatial feature extraction
        x = F.relu(self.spatial_conv1(x_reshaped))
        x = F.relu(self.spatial_conv2(x))
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        
        # Temporal feature extraction with LSTM
        lstm_out, (h_n, c_n) = self.temporal_lstm(x)
        x = self.layer_norm1(lstm_out[:, -1, :])  # Take last timestep
        
        # Self-attention for spatial relationships
        x_expanded = x.unsqueeze(1)  # (batch, 1, hidden_dim)
        attn_out, _ = self.attention(x_expanded, x_expanded, x_expanded)
        x = x + attn_out.squeeze(1)  # Residual connection
        x = self.layer_norm2(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for deep network optimization"""
    def __init__(self, dim: int):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.layer_norm(x + residual)  # Residual connection
        return F.relu(x)


class QNetwork3D(nn.Module):
    """
    3D Q-Network with optimized architecture
    Features:
    - Spatial-temporal encoder for 3D feature extraction
    - Residual blocks for deep learning
    - Dueling architecture (value + advantage streams)
    """
    def __init__(self, state_size: int, action_size: int, 
                 sequence_length: int = 4, hidden_dim: int = 128):
        super(QNetwork3D, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # 3D Spatial-Temporal Encoder
        self.encoder = SpatialTemporalEncoder(state_size, hidden_dim)
        
        # Dueling Architecture: Value and Advantage streams
        # Value stream (V(s))
        self.value_stream = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream (A(s,a))
        self.advantage_stream = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_size)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: Input state tensor
                - If 2D: (batch, features) -> converted to (batch, 1, features)
                - If 3D: (batch, seq_len, features)
        Returns:
            Q-values for all actions: (batch, action_size)
        """
        # Handle both 2D and 3D inputs
        if len(state.shape) == 2:
            state = state.unsqueeze(1)  # Add sequence dimension
        
        # Ensure sequence length matches
        if state.shape[1] < self.sequence_length:
            # Pad with zeros if needed
            padding = torch.zeros(state.shape[0], 
                                self.sequence_length - state.shape[1], 
                                state.shape[2], 
                                device=state.device)
            state = torch.cat([state, padding], dim=1)
        elif state.shape[1] > self.sequence_length:
            # Take last sequence_length timesteps
            state = state[:, -self.sequence_length:, :]
        
        # Encode spatial-temporal features
        encoded = self.encoder(state)
        encoded = self.dropout(encoded)
        
        # Dueling architecture
        value = self.value_stream(encoded)
        advantage = self.advantage_stream(encoded)
        
        # Combine value and advantage: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


# ============================================================================
# PRIORITIZED EXPERIENCE REPLAY BUFFER
# ============================================================================

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    Samples experiences based on TD-error priority
    """
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.0001
        self.max_beta = 1.0
        
        # Storage: (priority, experience)
        self.buffer = []
        self.priorities = []
        self.position = 0
        
        # Experience structure: (state, action, reward, next_state, done)
        self.experiences = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done, td_error: float = None):
        """
        Add experience to buffer with priority
        Args:
            td_error: TD-error for priority calculation (if None, uses max priority)
        """
        # Calculate priority from TD-error
        if td_error is None:
            priority = 1.0  # Max priority for new experiences
        else:
            priority = (abs(td_error) + 1e-6) ** self.alpha
        
        # Store experience
        experience = (state, action, reward, next_state, done)
        
        if len(self.experiences) < self.capacity:
            self.experiences.append(experience)
            self.priorities.append(priority)
        else:
            self.experiences[self.position] = experience
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample batch with importance sampling weights
        Returns:
            (states, actions, rewards, next_states, dones, indices, weights)
        """
        if len(self.experiences) < batch_size:
            return None
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities[:len(self.experiences)])
        probabilities = priorities / priorities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.experiences), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.experiences) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Sample experiences
        batch = [self.experiences[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Update beta
        self.beta = min(self.max_beta, self.beta + self.beta_increment)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            if idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.experiences)


# ============================================================================
# 3D DOUBLE DEEP Q-NETWORK AGENT
# ============================================================================

class DDQNAgent3D:
    """
    3D Double Deep Q-Network Agent with Prioritized Experience Replay
    """
    def __init__(self, state_size: int, action_size: int,
                 sequence_length: int = 4,
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 tau: float = 0.01,
                 batch_size: int = 16,
                 memory_size: int = 20000,
                 hidden_dim: int = 128,
                 device: str = None):
        """
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            sequence_length: Length of temporal sequence for 3D processing
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            tau: Soft update coefficient for target network
            batch_size: Batch size for training
            memory_size: Size of replay buffer
            device: Device to run on ('cuda' or 'cpu')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.batch_size = batch_size
        
        # Device setup
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dual Q-Networks
        self.q_network = QNetwork3D(state_size, action_size, sequence_length, hidden_dim=hidden_dim).to(self.device)
        self.target_network = QNetwork3D(state_size, action_size, sequence_length, hidden_dim=hidden_dim).to(self.device)
        
        # Initialize target network with main network weights
        self.update_target_network(hard_update=True)
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Prioritized Experience Replay
        self.memory = PrioritizedReplayBuffer(capacity=memory_size)
        
        # State history for 3D processing
        self.state_history = deque(maxlen=sequence_length)
        
        # Training statistics
        self.training_losses = []
        self.td_errors = []
        
    def update_target_network(self, hard_update: bool = False):
        """Update target network weights"""
        if hard_update:
            # Hard update: copy all weights
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            # Soft update: gradual weight transfer
            for target_param, local_param in zip(self.target_network.parameters(),
                                                self.q_network.parameters()):
                target_param.data.copy_(self.tau * local_param.data + 
                                      (1.0 - self.tau) * target_param.data)
    
    def act(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None,
            training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        Args:
            state: Current state
            action_mask: Mask for invalid actions (None = all valid)
            training: Whether in training mode (affects epsilon)
        Returns:
            Selected action index
        """
        # Add state to history
        self.state_history.append(state)
        
        # Epsilon-greedy exploration
        if training and random.random() <= self.epsilon:
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    return np.random.choice(valid_actions)
            return random.randrange(self.action_size)
        
        # Greedy action selection
        with torch.no_grad():
            # Prepare state sequence
            if len(self.state_history) < self.sequence_length:
                # Pad with current state if history is short
                state_sequence = list(self.state_history)
                while len(state_sequence) < self.sequence_length:
                    state_sequence.insert(0, state)
            else:
                state_sequence = list(self.state_history)
            
            state_tensor = torch.FloatTensor(np.array(state_sequence)).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy().flatten()
            
            # Apply action mask
            if action_mask is not None:
                q_values = np.where(action_mask, q_values, -np.inf)
            
            return np.argmax(q_values)
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool, td_error: float = None):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done, td_error)
    
    def replay(self) -> Optional[float]:
        """
        Train the network on a batch of experiences
        Returns:
            Average loss if training occurred, None otherwise
        """
        # Sample batch from prioritized replay buffer
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return None
        
        states, actions, rewards, next_states, dones, indices, weights = batch
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Prepare state sequences (pad if necessary)
        states_seq = self._prepare_sequence(states)
        next_states_seq = self._prepare_sequence(next_states)
        
        # Current Q-values
        current_q_values = self.q_network(states_seq).gather(1, actions.unsqueeze(1))
        
        # Double DQN: Use main network to select actions, target network to evaluate
        with torch.no_grad():
            # Select best actions using main network
            next_actions = self.q_network(next_states_seq).argmax(1)
            # Evaluate using target network
            next_q_values = self.target_network(next_states_seq).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Calculate TD-errors for priority updates
        td_errors = (target_q_values - current_q_values).detach().cpu().numpy().flatten()
        
        # Weighted loss (importance sampling)
        loss = (weights.unsqueeze(1) * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors)
        
        # Update target network (soft update)
        self.update_target_network(hard_update=False)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Store statistics
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        self.td_errors.extend(td_errors.tolist())
        
        return loss_value
    
    def _prepare_sequence(self, states: torch.Tensor) -> torch.Tensor:
        """Prepare state sequences for 3D processing"""
        batch_size = states.shape[0]
        
        # For now, duplicate states to create sequence (in real scenario, use history)
        # In practice, you would maintain state histories per sample
        if len(states.shape) == 2:
            # Expand to sequence: (batch, 1, features) -> (batch, seq_len, features)
            states = states.unsqueeze(1)
            states = states.repeat(1, self.sequence_length, 1)
        
        return states
    
    def save(self, filepath: str):
        """Save model weights"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)


# ============================================================================
# MULTI-AGENT 3D DDQN SYSTEM
# ============================================================================

class MultiAgentDDQN3D:
    """
    Multi-Agent 3D DDQN System for UAV Coordination
    """
    def __init__(self, state_size: int, action_size: int, num_agents: int,
                 sequence_length: int = 4, hidden_dim: int = 128, **kwargs):
        """
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            num_agents: Number of UAV agents
            sequence_length: Temporal sequence length for 3D processing
            **kwargs: Additional arguments for DDQNAgent3D
        """
        self.num_agents = num_agents
        self.agents = [
            DDQNAgent3D(state_size, action_size, sequence_length, hidden_dim=hidden_dim, **kwargs)
            for _ in range(num_agents)
        ]
    
    def act(self, states: List[np.ndarray], action_masks: List[np.ndarray] = None,
            training: bool = True) -> List[int]:
        """Get actions for all agents"""
        if action_masks is None:
            action_masks = [None] * self.num_agents
        
        return [
            agent.act(state, action_mask=mask, training=training)
            for agent, state, mask in zip(self.agents, states, action_masks)
        ]
    
    def remember(self, states: List[np.ndarray], actions: List[int],
                rewards: List[float], next_states: List[np.ndarray],
                dones: List[bool]):
        """Store experiences for all agents"""
        for i, agent in enumerate(self.agents):
            agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
    
    def replay(self):
        """Train all agents"""
        losses = []
        for agent in self.agents:
            loss = agent.replay()
            if loss is not None:
                losses.append(loss)
        return np.mean(losses) if losses else None
    
    def save_models(self, path_prefix: str):
        """Save all agent models"""
        for i, agent in enumerate(self.agents):
            agent.save(f"{path_prefix}_agent{i}.pth")
    
    def load_models(self, path_prefix: str):
        """Load all agent models"""
        for i, agent in enumerate(self.agents):
            agent.load(f"{path_prefix}_agent{i}.pth")


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_3d_ddqn(env, agent: MultiAgentDDQN3D, episodes: int = 1000,
                 update_frequency: int = 4, target_update_frequency: int = 100,
                 save_frequency: int = 100, save_path: str = "3d_ddqn_model"):
    """
    Training loop for 3D DDQN
    """
    episode_rewards = []
    episode_losses = []
    
    print("="*70)
    print("Training 3D Double Deep Q-Network for UAV-IoV Task Offloading")
    print("="*70)
    
    for episode in range(episodes):
        states = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done:
            # Get actions from all agents
            action_masks = [env.get_action_mask(i) for i in range(env.num_uavs)]
            actions = agent.act(states, action_masks=action_masks, training=True)
            
            # Execute actions
            next_states, rewards, dones, _ = env.step(actions)
            
            # Store experiences
            agent.remember(states, actions, rewards, next_states, dones)
            
            # Train agents periodically
            if step_count % update_frequency == 0 and len(agent.agents[0].memory) >= agent.agents[0].batch_size:
                loss = agent.replay()
                if loss is not None:
                    episode_losses.append(loss)
            
            episode_reward += np.mean(rewards)
            states = next_states
            done = all(dones)
            step_count += 1
        
        episode_rewards.append(episode_reward)
        
        # Progress reporting
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_loss = np.mean(episode_losses[-50:]) if episode_losses else 0
            epsilon = agent.agents[0].epsilon
            
            print(f"Episode {episode+1}/{episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Memory: {len(agent.agents[0].memory)}")
        
        # Save models periodically
        if (episode + 1) % save_frequency == 0:
            agent.save_models(f"{save_path}_ep{episode+1}")
            print(f"Models saved at episode {episode+1}")
    
    # Final save
    agent.save_models(f"{save_path}_final")
    print(f"\nTraining completed! Final models saved.")
    
    return episode_rewards, episode_losses


def evaluate_3d_ddqn(env, agent: MultiAgentDDQN3D, n_episodes: int = 20):
    """
    Evaluate trained 3D DDQN agent
    """
    test_rewards = []
    success_rates = []
    battery_levels = []
    
    print("\n" + "="*70)
    print("Evaluating 3D DDQN Agent")
    print("="*70)
    
    for episode in range(n_episodes):
        states = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action_masks = [env.get_action_mask(i) for i in range(env.num_uavs)]
            actions = agent.act(states, action_masks=action_masks, training=False)
            next_states, rewards, dones, _ = env.step(actions)
            episode_reward += np.mean(rewards)
            states = next_states
            done = all(dones)
        
        test_rewards.append(episode_reward)
        success_rates.append(np.mean([1 if uav['current_task'] is None else 0 
                                     for uav in env.uavs]))
        battery_levels.append(np.mean([uav['battery'] for uav in env.uavs]))
    
    # Print results
    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"Average Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Task Success Rate: {np.mean(success_rates):.2%}")
    print(f"Average Battery Level: {np.mean(battery_levels):.1f}%")
    print(f"Best Episode Reward: {max(test_rewards):.2f}")
    
    return {
        'rewards': test_rewards,
        'success_rates': success_rates,
        'battery_levels': battery_levels
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Import environment (assuming it exists)
    import importlib.util
    try:
        from ddqfinaltraining import UAVIoVEnv
    except ImportError:
        try:
            # Handle files with hyphens in name
            spec = importlib.util.spec_from_file_location(
                "security_offloading", 
                "Security-AwareTrust-BasedOffloading.py"
            )
            security_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(security_module)
            UAVIoVEnv = security_module.UAVIoVEnv
        except Exception:
            print("Warning: Could not import UAVIoVEnv. Please ensure the environment is available.")
            print("This file provides the 3D DDQN architecture components.")
            print("To use, import DDQNAgent3D or MultiAgentDDQN3D in your training script.")
            exit(0)
    
    # Create environment
    env = UAVIoVEnv(enable_security=False)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create 3D DDQN agent
    agent = MultiAgentDDQN3D(
        state_size=state_size,
        action_size=action_size,
        num_agents=env.num_uavs,
        sequence_length=4,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        tau=0.01,
        batch_size=64,
        memory_size=100000
    )
    
    # Train
    print("\nStarting training...")
    rewards, losses = train_3d_ddqn(
        env=env,
        agent=agent,
        episodes=1000,
        update_frequency=4,
        save_frequency=100,
        save_path="3d_ddqn_uav_iov"
    )
    
    # Evaluate
    results = evaluate_3d_ddqn(env, agent, n_episodes=20)
    
    print("\n" + "="*70)
    print("Training and Evaluation Complete!")
    print("="*70)
