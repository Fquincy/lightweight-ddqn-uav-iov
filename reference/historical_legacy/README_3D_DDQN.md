# 3D Double Deep Q-Network (DDQN) for UAV-IoV Task Offloading

## Quick Start

### Installation

Ensure you have the required dependencies:
```bash
pip install torch numpy matplotlib gym
```

### Files Overview

- **`ddqn_3d_uav_iov.py`**: Core 3D DDQN architecture implementation
- **`train_3d_ddqn.py`**: Complete training script
- **`test_3d_ddqn.py`**: Test script to verify architecture
- **`DDQN_3D_ARCHITECTURE.md`**: Detailed architecture documentation

### Quick Test

Verify the architecture works:
```bash
python test_3d_ddqn.py
```

### Training

Train the 3D DDQN agent:
```bash
python train_3d_ddqn.py
```

## Architecture Highlights

### ✨ Key Features

1. **3D Spatial-Temporal Processing**
   - Processes sequences of states over time
   - Captures spatial relationships between UAVs, vehicles, and edge servers
   - Uses 1D convolutions + LSTM + attention mechanisms

2. **Prioritized Experience Replay**
   - Samples important experiences more frequently
   - Uses TD-error based prioritization
   - Includes importance sampling to correct bias

3. **Dual Q-Networks**
   - Main network for action selection
   - Target network for value estimation
   - Soft updates for stable learning

4. **Optimized Network Design**
   - Dueling architecture (value + advantage streams)
   - Residual connections for deep networks
   - Multi-head attention for spatial relationships
   - Layer normalization for stable training

### Architecture Components

```
Input (State Sequence)
    ↓
Spatial-Temporal Encoder
    ├─ 1D Convolutions (Spatial Features)
    ├─ LSTM (Temporal Features)
    └─ Multi-Head Attention (Spatial Relationships)
    ↓
Residual Blocks (Deep Processing)
    ↓
Dueling Architecture
    ├─ Value Stream → V(s)
    └─ Advantage Stream → A(s,a)
    ↓
Q-Values: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
```

## Usage Example

```python
from ddqn_3d_uav_iov import MultiAgentDDQN3D, train_3d_ddqn, evaluate_3d_ddqn
from ddqfinaltraining import UAVIoVEnv

# Create environment
env = UAVIoVEnv()

# Create 3D DDQN agent
agent = MultiAgentDDQN3D(
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.n,
    num_agents=env.num_uavs,
    sequence_length=4,  # Temporal sequence length
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
rewards, losses = train_3d_ddqn(
    env=env,
    agent=agent,
    episodes=1000,
    update_frequency=4,
    save_frequency=100,
    save_path="3d_ddqn_model"
)

# Evaluate
results = evaluate_3d_ddqn(env, agent, n_episodes=20)
```

## Hyperparameters

### Recommended Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `sequence_length` | 4 | Temporal sequence length |
| `hidden_dim` | 256 | Hidden layer dimension |
| `lr` | 0.001 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `epsilon` | 1.0 | Initial exploration |
| `epsilon_min` | 0.01 | Minimum exploration |
| `epsilon_decay` | 0.995 | Exploration decay |
| `tau` | 0.01 | Soft update coefficient |
| `batch_size` | 64 | Training batch size |
| `memory_size` | 100000 | Replay buffer size |

## Advantages

### vs Standard DQN
- ✅ Better spatial-temporal feature extraction
- ✅ Faster learning with prioritized replay
- ✅ Reduced overestimation bias (Double DQN)
- ✅ More expressive with dueling architecture

### vs Standard DDQN
- ✅ 3D processing captures temporal patterns
- ✅ Attention mechanisms focus on important relationships
- ✅ Residual connections enable deeper networks
- ✅ Prioritized replay improves sample efficiency

## Performance Tips

1. **GPU Acceleration**: The architecture automatically uses GPU if available
2. **Batch Size**: Increase batch size for faster training (if memory allows)
3. **Sequence Length**: Adjust based on temporal dependencies in your task
4. **Memory Size**: Larger buffer = more diverse experiences but slower sampling

## Troubleshooting

### Import Errors
If you get import errors for the environment:
- Ensure `ddqfinaltraining.py` or `Security-AwareTrust-BasedOffloading.py` exists
- Check Python path includes the project directory

### Memory Issues
- Reduce `batch_size` or `memory_size`
- Reduce `sequence_length`
- Use CPU instead of GPU if GPU memory is limited

### Training Instability
- Reduce learning rate (`lr`)
- Increase `tau` for slower target network updates
- Check gradient clipping is enabled (it is by default)

## Citation

If you use this architecture in your research, please cite:

```bibtex
@software{ddqn_3d_uav_iov,
  title={3D Double Deep Q-Network for UAV-IoV Task Offloading},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## License

This implementation is provided for research and educational purposes.

## Contact

For questions or issues, please open an issue on the repository.
