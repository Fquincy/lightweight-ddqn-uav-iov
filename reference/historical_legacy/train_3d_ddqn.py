"""
Training Script for 3D Double Deep Q-Network (DDQN) for UAV-IoV Task Offloading
================================================================================
This script trains a 3D DDQN agent with prioritized experience replay
on the UAV-IoV task offloading environment.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
import importlib.util

# Keep CPU usage lighter to avoid memory spikes
torch.set_num_threads(1)

# Ensure current directory is in Python path
if '.' not in sys.path:
    sys.path.insert(0, '.')

# Import the 3D DDQN architecture
from ddqn_3d_uav_iov import MultiAgentDDQN3D, train_3d_ddqn, evaluate_3d_ddqn

# Import environment (try both possible names)
try:
    from ddqfinaltraining import UAVIoVEnv
except ImportError as e:
    try:
        # Handle files with hyphens in name
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "security_offloading", 
            "Security-AwareTrust-BasedOffloading.py"
        )
        security_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(security_module)
        UAVIoVEnv = security_module.UAVIoVEnv
    except Exception as e:
        print(f"Error: Could not import UAVIoVEnv. {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("\nPlease ensure the environment file is available and all dependencies are installed.")
        print("Required packages: numpy, torch, matplotlib, gym, pandas")
        sys.exit(1)


def plot_training_results(rewards, losses, save_path="training_results.png"):
    """Plot training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(rewards, alpha=0.3, label='Episode Reward')
    if len(rewards) >= 10:
        window = min(50, len(rewards) // 10)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(rewards)), moving_avg, 
                       label=f'{window}-Episode Average', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training losses
    if losses:
        axes[0, 1].plot(losses, alpha=0.5, label='Training Loss')
        if len(losses) >= 10:
            window = min(50, len(losses) // 10)
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(losses)), moving_avg,
                           label=f'{window}-Step Average', linewidth=2)
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Reward distribution
    axes[1, 0].hist(rewards, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(rewards), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(rewards):.2f}')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning curve (smoothed)
    if len(rewards) >= 20:
        window = 20
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[1, 1].plot(range(window-1, len(rewards)), smoothed, linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Smoothed Reward')
        axes[1, 1].set_title('Learning Curve (20-Episode Moving Average)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining plots saved to {save_path}")
    plt.close()


def main():
    """Main training function"""
    print("="*70)
    print("3D Double Deep Q-Network (DDQN) Training")
    print("UAV-IoV Task Offloading with Prioritized Experience Replay")
    print("="*70)
    
    # Create environment
    print("\n[1/5] Initializing environment...")
    env = UAVIoVEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    num_agents = env.num_uavs
    
    print(f"  State size: {state_size}")
    print(f"  Action size: {action_size}")
    print(f"  Number of agents: {num_agents}")
    
    # Create 3D DDQN agent
    print("\n[2/5] Creating 3D DDQN agent...")
    agent = MultiAgentDDQN3D(
        state_size=state_size,
        action_size=action_size,
        num_agents=num_agents,
        sequence_length=4,  # Temporal sequence length for 3D processing
        hidden_dim=128,     # Reduced model width to save memory
        lr=0.001,           # Learning rate
        gamma=0.99,         # Discount factor
        epsilon=1.0,        # Initial exploration rate
        epsilon_min=0.01,   # Minimum exploration rate
        epsilon_decay=0.995, # Epsilon decay rate
        tau=0.01,           # Soft update coefficient
        batch_size=16,      # Smaller batch to reduce RAM
        memory_size=20000   # Smaller replay buffer to save RAM
    )
    
    print("  [OK] 3D Spatial-Temporal Encoder initialized")
    print("  [OK] Dual Q-Networks (Main & Target) created")
    print("  [OK] Prioritized Experience Replay buffer initialized")
    print("  [OK] Multi-agent system ready")
    
    # Training parameters
    episodes = 300          # Lighter run to avoid long high-RAM sessions
    update_frequency = 4
    save_frequency = 100
    
    print(f"\n[3/5] Training configuration:")
    print(f"  Episodes: {episodes}")
    print(f"  Update frequency: Every {update_frequency} steps")
    print(f"  Save frequency: Every {save_frequency} episodes")
    
    # Train agent
    print("\n[4/5] Starting training...")
    print("-"*70)
    
    try:
        rewards, losses = train_3d_ddqn(
            env=env,
            agent=agent,
            episodes=episodes,
            update_frequency=update_frequency,
            target_update_frequency=100,
            save_frequency=save_frequency,
            save_path="3d_ddqn_uav_iov"
        )
        
        print("-"*70)
        print("\n[5/5] Training completed successfully!")
        
        # Plot results
        print("\nGenerating training plots...")
        plot_training_results(rewards, losses, "3d_ddqn_training_results.png")
        
        # Evaluate agent
        print("\nEvaluating trained agent...")
        results = evaluate_3d_ddqn(env, agent, n_episodes=20)
        
        # Print summary
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        print(f"Final Average Reward: {np.mean(rewards[-50:]):.2f}")
        print(f"Best Episode Reward: {max(rewards):.2f}")
        print(f"Evaluation Success Rate: {np.mean(results['success_rates']):.2%}")
        print(f"Evaluation Average Battery: {np.mean(results['battery_levels']):.1f}%")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving models...")
        try:
            agent.save_models("3d_ddqn_uav_iov_interrupted")
            print("Models saved successfully.")
        except Exception as save_err:
            print(f"Warning: could not save models on interrupt: {save_err}")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        print("\nAttempting to save models before exit...")
        try:
            agent.save_models("3d_ddqn_uav_iov_error")
            print("Models saved.")
        except Exception as save_err:
            print(f"Warning: could not save models after error: {save_err}")
        raise


if __name__ == "__main__":
    main()
