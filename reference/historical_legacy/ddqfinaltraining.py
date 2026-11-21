import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
plt.ioff()
from matplotlib import cm
import gym
from gym import spaces
import pandas as pd
from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore', message='Box bound precision lowered')
import gc

plt.rcParams['figure.max_open_warning'] = 30
plt.style.use('ggplot')

class UAVIoVEnv(gym.Env):
    def __init__(self, eval_attack_extra=0.0):
        super(UAVIoVEnv, self).__init__()
        self.eval_attack_extra = eval_attack_extra  # Extra failure prob for "under attack" eval
        # System configuration
        self.task_reward_multiplier = 1.0  # Add this EXACT line
        self.num_vehicles = 10
        self.num_uavs = 2
        self.num_edge_servers = 2
        self.max_steps = 200

        # Charging stations with faster charging
        self.charging_stations = [
            {'location': np.array([20, 20]), 'rate': 0.12, 'energy_saved': 0},
            {'location': np.array([80, 80]), 'rate': 0.12, 'energy_saved': 0}
   ]
        self.cloud_server = {'location': np.array([50, 50]), 'latency': 50}

        self.uavs = [{
            'battery': 80 + random.uniform(-5, 5),
            'location': np.array([20 + random.uniform(-5, 5),
                                  20 + random.uniform(-5, 5)]),
            'charging_profile': {
                'min_threshold': 40 + i*10,  # UAV0:40%, UAV1:50%
                'max_threshold': 90 - i*5    # UAV0:90%, UAV1:85%
            },
            'is_charging': False
            } for i in range(self.num_uavs)]
        
        self.vehicles = [{
            'location': np.random.uniform(0, 100, size=2),
            'battery': np.random.uniform(30, 100),
            'task': {
                'size': np.random.uniform(10, 300),
                'priority': random.randint(0, 2)
            } if random.random() < 0.7 else None
        } for _ in range(self.num_vehicles)]

        self.edge_servers = [{
            'location': np.random.uniform(0, 100, size=2),
            'capacity': np.random.uniform(100, 500)
        } for _ in range(self.num_edge_servers)]

        self.cloud_server = {'location': np.array([50, 50]), 'latency': 50}

        # Dynamic discharge rates
        self.energy_params = {
            'idle_discharge': 0.08,  # Reduced from 0.15
            'active_discharge': 0.15, # Reduced from 0.25 
            'move_cost': 0.05,       # Reduced from 0.1
            'compute_cost': 0.02     # Reduced from 0.05
       }

        # Observation space
        self.observation_space = spaces.Box(
            low=np.zeros(10),
            high=np.ones(10),
            dtype=np.float32
        )

        # Action space
        self.action_space = spaces.Discrete(5)
        
        # Initialize components
        self.reset()
        
        # Visualization setup
        self.fig = plt.figure(figsize=(18, 8))
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        
        # Metrics tracking
        self.metrics = {
            'episode_rewards': [],
            'avg_latency': [],
            'task_completion_rate': [],  # Add this line
            'energy_usage': [],
            'action_counts': np.zeros(5),
            'uav_batteries': [[] for _ in range(self.num_uavs)],
            'energy_saved': [],
            'success_rate': [],
            'avg_battery': [],
            'uav_utilization': []
        }

    def get_action_mask(self, uav_index):
        uav = self.uavs[uav_index]
        mask = [1] * 5  # Default all actions allowed
        
        # Prevent charging when not needed
        nearest_dist = min(np.linalg.norm(uav['location'] - s['location'])
                          for s in self.charging_stations)
        if nearest_dist > 10:
            mask[4] = 0

        charging_uavs = sum(1 for u in self.uavs if u['is_charging'])
        if charging_uavs >= 1 and not uav['is_charging']:
            mask[4] = 0 

        if charging_uavs >= 1 and uav['battery'] > 50:
            mask[4] = 0

        should_charge = (
            uav['battery'] < 30 and
            nearest_dist <= 10 and
            charging_uavs == 0
       )

        if should_charge:
            mask = [0,0,0,0,1]
        elif uav['current_task'] and uav['battery'] > 40:
            mask[4] = 0

        if uav['cpu_load'] > 70:
            mask[0] = 0

        return mask

    def reset(self):
        self.vehicles = [{
            'location': np.random.uniform(0, 100, size=2),
            'battery': np.random.uniform(30, 100),
            'task': {
                'size': np.random.uniform(10, 300),
                'priority': random.randint(0, 2)
            } if random.random() < 0.7 else None
        } for _ in range(self.num_vehicles)]
        
        self.uavs = [{
            'battery': 80 + random.uniform(-5, 5),
            'location': np.array([
                20 + random.uniform(-5, 5) if idx == 0 else 80 + random.uniform(-5, 5),
                20 + random.uniform(-5, 5) if idx == 0 else 80 + random.uniform(-5, 5)
            ]),
            'charging_profile': {
                'min_threshold': 40 + idx*10,
                'max_threshold': 90 - idx*5 
            },
            'is_charging': False,

            'prev_location': np.array([0, 0]),
            'cpu_load': 30 + random.uniform(0, 10),
            'current_task': None,
            'is_charging': False,
            'last_action': None,
            'should_charge': False,
            'active_steps': 0
        } for idx in range(self.num_uavs)]
        
        self.edge_servers = [{
            'location': np.array([20 + i*60, 80 - i*60]),
            'load': random.uniform(0, 50)
        } for i in range(self.num_edge_servers)]
        
        self.current_step = 0
        self._generate_tasks()
        return self._get_observations()
    
    def _get_observations(self):
        return [self._get_uav_observation(uav) for uav in self.uavs]
    
    def _get_uav_observation(self, uav):
        nearest_charge_dist = min(
            np.linalg.norm(uav['location'] - s['location'])
            for s in self.charging_stations
        )
        
        active_tasks = len([v for v in self.vehicles if v['task'] is not None])
        
        return np.array([
            uav['battery'] / 100,
            (uav['current_task']['size'] if uav['current_task'] else 10) / 300,
            (np.random.uniform(100, 1000) - 100) / 900,
            uav['cpu_load'] / 100,
            (uav['current_task']['priority'] if uav['current_task'] else 0) / 2,
            float(uav['is_charging']),
            nearest_charge_dist / 100,
            float(uav['battery'] < 20),
            self.current_step / self.max_steps,
            active_tasks / self.num_vehicles
        ], dtype=np.float32)
    
    # Gradually increase task difficulty
    def _generate_tasks(self):
        progress = min(1, self.current_step/self.max_steps)
        base_prob = 0.5 + 0.3*progress  # Ramp from 50% to 80%
    
        for v in self.vehicles:
            if random.random() < base_prob:
                v['task'] = {
                    'size': np.random.uniform(10, 200 + 100*progress),
                    'priority': random.choices([0,1,2], 
                         weights=[0.7-0.2*progress, 
                                 0.2+0.1*progress, 
                                 0.1+0.1*progress])[0]
            }
    
    def step(self, actions):
        assert len(actions) == self.num_uavs
        rewards = [0] * self.num_uavs
        dones = [False] * self.num_uavs
        infos = [{} for _ in range(self.num_uavs)]
        energy_saved_this_step = 0

        for uav in self.uavs:
            if 'charging_profile' not in uav:
                uav['charging_profile'] = {
                    'min_threshold': 40,
                    'max_threshold': 90
                }
                
            if uav['is_charging'] and uav['battery'] > uav['charging_profile']['max_threshold']:
                uav['is_charging'] = False
                uav['battery'] = uav['charging_profile']['max_threshold']

        for i, (uav, action) in enumerate(zip(self.uavs, actions)):
            uav['last_action'] = action
            uav['prev_location'] = np.copy(uav['location'])
            latency = 0
            energy = 0
            task_completed = False

            discharge_rate = self.energy_params['active_discharge'] if uav['current_task'] else self.energy_params['idle_discharge']
            uav['battery'] = max(0, uav['battery'] - discharge_rate)

            for uav in self.uavs:
                if uav['battery'] >= 99.9:  # Hard cap battery
                    uav['battery'] = 99.9
                    uav['is_charging'] = False

            if uav['current_task'] is not None:
                latency, energy = self._process_task(uav, action)
                infos[i]['task_completed'] = latency <= 100
                infos[i]['latency'] = latency if latency != float('inf') else 200.0
                infos[i]['energy'] = energy
                task_completed = latency <= 100

            if abs(self.uavs[0]['battery'] - self.uavs[1]['battery']) > 30:
                rewards[i] -= 1

            if action == 4:  # Charging
                nearest_station = min(self.charging_stations,
                        key=lambda s: np.linalg.norm(uav['location'] - s['location']))
                dist = np.linalg.norm(uav['location'] - nearest_station['location'])
                
                if dist < 10:
                    charge_rate = nearest_station['rate'] * (1 + (1 - uav['battery']/100))
                    
                    if uav['battery'] > 60:
                        saved_energy = nearest_station['rate'] * 0.3
                        charge_rate *= 0.7
                    if 'energy_saved' not in nearest_station:
                        nearest_station['energy_saved'] = 0
                        nearest_station['energy_saved'] += saved_energy
                    else:
                       conservation_bonus = 0
                    
                    uav['battery'] = min(100, uav['battery'] + charge_rate * 100)
                    uav['is_charging'] = True
                    rewards[i] += 1.0 * (charge_rate * 100)
                else:
                    uav['is_charging'] = False
                    rewards[i] -= 1
            else:
                uav['is_charging'] = False
                
                if uav['current_task'] is None:
                    rewards[i] -= 1
                else:
                    rewards[i] += self._calculate_reward(uav, action, latency, energy)
                    
                    if task_completed:
                        priority_bonus = 2 * (1 + uav['current_task']['priority'])
                        rewards[i] += priority_bonus

            # Enforce single-charger policy
            active_chargers = [u for u in self.uavs if u['is_charging']]
            if len(active_chargers) > 1:
                for u in active_chargers[1:]:  # Only allow first charger
                    u['is_charging'] = False
                    u['battery'] = min(99.9, u['battery'])  # Hard cap

            self.metrics['action_counts'][action] += 1
            self.metrics['energy_usage'].append(energy)
            uav['active_steps'] += 1

        self.metrics['energy_saved'].append(energy_saved_this_step)
        self.metrics['success_rate'].append(
            sum(1 for uav in self.uavs if uav['current_task'] is None)/self.num_uavs)
        self.metrics['avg_battery'].append(
            sum(uav['battery'] for uav in self.uavs)/self.num_uavs)
        self.metrics['uav_utilization'].append(
            sum(uav['cpu_load'] for uav in self.uavs)/self.num_uavs)
        
        rewards = np.clip(rewards, -10, 10).tolist()
        
        self.current_step += 1
        done = self.current_step >= self.max_steps or all(uav['battery'] <= 0 for uav in self.uavs)
        dones = [done] * self.num_uavs
        
        completed_tasks = sum(1 for uav in self.uavs if uav['current_task'] is None)
        self.metrics['task_completion_rate'].append(completed_tasks / self.num_uavs)
        self._generate_tasks()
        observations = self._get_observations()

        return observations, rewards, dones, {'agent_infos': infos}

    def _process_task(self, uav, action):
        task = uav['current_task']
        latency = 0
        energy = 0

        if uav['is_charging'] or any(u['is_charging'] for u in self.uavs if u != uav):
            return float('inf'), 0
        
        if uav['battery'] < 15:
           return float('inf'), 0  # Fail task if critically low

        # Eval attack mode: extra failure prob to simulate malicious/unreliable nodes
        if getattr(self, 'eval_attack_extra', 0) > 0 and random.random() < self.eval_attack_extra:
            return float('inf'), 0

        # Success probability based on battery and CPU
        success_prob = 0.7 + 0.2 * (uav['battery']/100)  # Battery affects success
        if random.random() > success_prob:
            return float('inf'), 0  # Task failed
        
        # Modified energy costs - ONLY CHANGE THESE LINES
        energy_costs = {
            0: 0.25,  # Local (was 0.15)
            1: 0.08,   # UAV (was 0.04)
            2: 0.05,   # Edge (was 0.025)
            3: 0.25    # Cloud (was 0.21)
        }
        energy = task['size'] * (0.4 if action != 3 else 0.3)

        if action == 0:
            latency = max(1, task['size'] / (110 - uav['cpu_load']) * 3)
            energy = task['size'] * energy_costs[0]
            uav['cpu_load'] = min(100, uav['cpu_load'] + task['size'] * 0.12)
        elif action == 1:
            latency = task['size'] / 500 * 100 + 10
            energy = task['size'] * energy_costs[1]
        elif action == 2:
            nearest_edge = min(self.edge_servers,
                             key=lambda e: np.linalg.norm(uav['location'] - e['location']))
            latency = task['size'] / 800 * 150 + 15 + nearest_edge['load']
            energy = task['size'] * energy_costs[2]
        elif action == 3:
            latency = task['size'] / 1000 * 200 + self.cloud_server['latency']
            energy = task['size'] * energy_costs[3]

        uav['cpu_load'] = max(0, uav['cpu_load'] * 0.9)
        uav['battery'] = max(0, uav['battery'] - energy * 3.0)
        
        return latency, energy
    
    def _calculate_reward(self, uav, action, latency, energy_cost):
        task = uav['current_task']
        battery_ratio = uav['battery'] / 100
        if task is None:
            return -0.5

        battery_penalty = (1 - battery_ratio) ** 2 * 3
        critical_penalty = 15.0 if uav['battery'] < 15 else 0 # Stronger penalty
        battery_ratio = uav['battery'] / 100

        # Initialize all reward components
        task_reward = 0
        completion_bonus = 0
        battery_penalty = 0
        battery_bonus = 0  # Initialize here
        charging_bonus = 0
        idle_penalty = 0
        conservation_bonus = 0
        nearest_dist = float('inf')  # Initialize with a large default value
        completion_bonus = 0  # Initialize completion_bonus here

        if uav['battery'] > 80 and action != 4:  # Penalize high battery without charging
            battery_penalty = (uav['battery'] - 80) * 0.5

        # Enhanced task rewards (priority-based)
        if task is not None:
            if latency <= 100:  # Task completed successfully
                base_reward = 20  # Increased base reward
                size_bonus = task['size'] / 50  # More weight to task size
                priority_bonus = task['priority'] * 10  # Triple priority importance
                time_bonus = (1 - latency/100) * 15  # Faster completion = higher reward
                task_reward = base_reward + size_bonus + priority_bonus + time_bonus
                completion_bonus = 25  # Significant bonus for completion
        else:
            task_reward = -15 * (latency / 100)  # Stronger penalty for failure
            idle_penalty = 2  # Penalize idle time more

        # Base success reward (dramatically increased)
        if latency <= 100 and uav['current_task']:
            priority = uav['current_task']['priority']
            success_reward = 10 * (1 + priority)  # 10-30 points per completed task
            latency_ratio = (100 - latency)/100
            speed_bonus += 5 * latency_ratio
            return success_reward + speed_bonus
        
        elif uav['current_task']:
            success_reward = -5 * (1 + uav['current_task']['priority'])
            return -10 * (1 + uav['current_task']['priority'])

        # Battery management (non-linear penalties)
        battery_ratio = uav['battery'] / 100
        if battery_ratio < 0.3:  # Only penalize very low battery
            battery_penalty = 8 * (0.3 - battery_ratio)**2  # Quadratic penalty
        elif battery_ratio > 0.8:  # Reward high battery
            charging_bonus = 3  # Small bonus for high battery

        # Charging rewards (only when necessary)
        if action == 4:  # Charging
            if uav['battery'] < 30:
                nearest_dist = min(np.linalg.norm(uav['location'] - s['location']) 
                                   for s in self.charging_stations)
                if nearest_dist < 10:
                    return 1.0 
                return -5.0
        
        # Near charging stations
        if action == 4 and nearest_dist < 10:
            charge_reward = min(2, (100 - uav['battery'])*0.05)  # Max 2 reward points
            return charge_reward  # Override other rewards when charging
        
        if nearest_dist < 10:
            # Scale charging reward by battery need
            charging_bonus = min(2.0, (100 - uav['battery']) * 0.05)  # Max 2 points
            if uav['battery'] > 50:  # Penalize unnecessary charging
                charging_bonus *= 0.3 
                return charging_bonus

        # Energy conservation bonuses
        if action != 4 and energy_cost < 0.5:
            conservation_bonus = 5 * (0.5 - energy_cost)
            return (task_reward + completion_bonus + charging_bonus + conservation_bonus
                    - battery_penalty - critical_penalty)

        # Task completion (higher priority)    
        if task:
            if latency <= 100:
                base_reward = 30  # Increased base reward
                priority_multiplier = 1 + task['priority'] * 0.5
                task_reward = base_reward * priority_multiplier
                completion_bonus = 20 * (1 - latency/100)  # Speed bonus  # Faster completion = higher bonus
        else:
            task_reward = -20  # Fixed penalty for failure  
            completion_bonus = 0

        return (task_reward + completion_bonus + battery_bonus + charging_bonus 
            - battery_penalty - idle_penalty)

    def render(self, mode='human'):
        clear_output(wait=True)
        
        if mode == 'human':
            plt.close('all')
            fig = plt.figure(figsize=(18, 8))
            
            # System view
            ax1 = fig.add_subplot(131)
            for station in self.charging_stations:
                ax1.plot(station['location'][0], station['location'][1], 'g^', markersize=15, 
                        label=f"Station (Saved: {station['energy_saved']:.1f}%)")
            
            for v in self.vehicles:
                color = 'r' if v['task'] else 'b'
                ax1.plot(v['location'][0], v['location'][1], f'{color}o', alpha=0.6)
            
            for i, uav in enumerate(self.uavs):
                ax1.plot(uav['location'][0], uav['location'][1], 
                        'ks' if i == 0 else 'kD', markersize=12,
                        label=f'UAV {i+1} ({uav["battery"]:.1f}%)')
            
            for es in self.edge_servers:
                ax1.plot(es['location'][0], es['location'][1], 'mx', markersize=12)
            
            ax1.plot(self.cloud_server['location'][0], self.cloud_server['location'][1],
                    'c*', markersize=15, label='Cloud')
            ax1.legend()
            ax1.set_title('System Overview')

            # Success rate heatmap
            success_grid = np.zeros((10,10))
            for x in range(10):
                for y in range(10):
                    success_grid[x,y] = ...
                    self.ax2.imshow(success_grid, cmap='viridis')
            
            # Enhanced Metrics
            ax2 = fig.add_subplot(132)
            if self.metrics['episode_rewards']:
                episodes = range(len(self.metrics['episode_rewards']))
                ax2.plot(episodes, self.metrics['episode_rewards'], 'b-', alpha=0.3, label='Raw Reward')
                ax2.plot(episodes, pd.Series(self.metrics['episode_rewards']).rolling(10).mean(), 
                        'r-', label='10-ep Avg')
                ax2.plot(episodes, pd.Series(self.metrics['episode_rewards']).rolling(50).mean(), 
                        'g-', label='50-ep Avg')
                
                # Only plot energy saved if we have enough data
                if len(self.metrics['energy_saved']) >= 10:
                    energy_episodes = range(len(self.metrics['energy_saved']))
                    ax2_twin = ax2.twinx()
                    ax2_twin.plot(energy_episodes, pd.Series(self.metrics['energy_saved']).rolling(10).mean(), 
                                 'm--', label='Energy Saved')
                    ax2_twin.set_ylabel('Energy Saved (%)')
                
                ax2.set_title('Training Performance')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Reward')
                ax2.legend(loc='upper left')
            
            # Action Distribution
            ax3 = fig.add_subplot(133)
            actions = ['Local', 'UAV', 'Edge', 'Cloud', 'Charge']
            total_actions = sum(self.metrics['action_counts'])
            
            if total_actions > 0:
                percentages = 100 * self.metrics['action_counts'] / total_actions
                bars = ax3.bar(actions, percentages, color=['blue', 'green', 'red', 'purple', 'orange'])
                ax3.set_title('Action Distribution')
                ax3.set_ylabel('Percentage (%)')
                
                ax3.text(0.5, -0.2, 
                        f"Total Energy Saved: {sum(self.metrics['energy_saved']):.1f}%",
                        ha='center', va='center', transform=ax3.transAxes)
                
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer()
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.batch_size = 512
        self.tau = 0.005
        self.recent_rewards = deque(maxlen=10)
        
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.update_target_model()

    def update_epsilon(self):
        decay = self.epsilon_decay
        if len(self.recent_rewards) >= 10:
            mean_reward = np.mean(self.recent_rewards)
            if mean_reward > 1e-8:
                improvement = (max(self.recent_rewards) - min(self.recent_rewards)) / mean_reward
                decay = max(0.99, min(0.999, 1 - improvement/10))
        return decay

    def act(self, state, action_mask=None):
        if random.random() <= self.epsilon:
            if action_mask is not None:
                valid_actions = [a for a, m in enumerate(action_mask) if m]
                return random.choice(valid_actions) if valid_actions else 0
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state).numpy().flatten()
            if action_mask is not None:
                q_values = np.where(action_mask, q_values, -np.inf)
            return np.argmax(q_values)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        samples = self.memory.sample(self.batch_size)
        if samples is None:
            return

        states, actions, rewards, next_states, dones = samples
        self.recent_rewards.extend(rewards)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_actions = self.model(next_states).argmax(1)
        next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))
        target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.update_epsilon()
    
    def soft_update(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

class MultiAgentDQN:
    def __init__(self, state_size, action_size, num_agents, env):
        self.env = env
        self.agents = [DoubleDQNAgent(state_size, action_size) for _ in range(num_agents)]
    
    def act(self, states):
        masks = [self.env.get_action_mask(i) for i in range(len(self.agents))]
        return [
            agent.act(state, action_mask=mask)
            for agent, state, mask in zip(self.agents, states, masks)
        ]
    
    def remember(self, states, actions, rewards, next_states, dones):
        for i, agent in enumerate(self.agents):
            agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
    
    def replay(self):
        for agent in self.agents:
            agent.replay()
    
    def soft_update(self):
        for agent in self.agents:
            agent.soft_update()
    
    def save_models(self, path_prefix):
        for i, agent in enumerate(self.agents):
            torch.save(agent.model.state_dict(), f"{path_prefix}_agent{i}.pth")
    
    def load_models(self, path_prefix):
        for i, agent in enumerate(self.agents):
            agent.model.load_state_dict(torch.load(f"{path_prefix}_agent{i}.pth"))
            agent.update_target_model()

def train_system(env, episodes=1000, save_path="uav_model"):
    try:
        agent = MultiAgentDQN(
            state_size=10,
            action_size=env.action_space.n,
            num_agents=env.num_uavs,
            env=env
        )
        
        # Initialize best_reward with negative infinity
        best_reward = -float('inf')  # or -np.inf if numpy is imported
        
        for ep in range(episodes):
            states = env.reset()
            episode_reward = 0
            done = False
            
            # Dynamic success threshold (70-80% target)
            target_success = 0.7 + 0.1 * (ep/episodes)

            # Adaptive battery thresholds
            min_success = 0.3 + (0.5 * (ep/episodes))  # Ramp from 30% to 80%
            min_battery = max(15, 40 - (ep // 40))
            
            for uav in env.uavs:
                uav['should_charge'] = uav['battery'] < min_battery
            
            while not done:
                actions = agent.act(states)
                next_states, rewards, dones, _ = env.step(actions)
                
                agent.remember(states, actions, rewards, next_states, dones)
                episode_reward += np.mean(rewards)
                states = next_states
                done = all(dones)

            # Get the average task completion rate for this episode
            if len(env.metrics['task_completion_rate']) > 10:
                avg_completion = np.mean(env.metrics['task_completion_rate'][-env.max_steps:][-10:])
            else:
                avg_completion = 0

            # Dynamic success threshold adjustment
            min_success = 0.3 + (0.5 * (ep/episodes)) if episodes > 0 else 0.3
            if avg_completion < min_success:
                pass

            if avg_completion > target_success:
                env.task_reward_multiplier *= 1.1
            else:
                env.task_reward_multiplier *= 1.1

            # Update best reward tracking
            if episode_reward > best_reward:
                best_reward = episode_reward
            
            # Experience replay and soft updates
            agent.replay()
            agent.soft_update()
            
            # Save only at episode 950 and 1000
            if ep == 950:
                agent.save_models(save_path + "_ep950")
                print(f"Saved model at episode 950 | Reward: {episode_reward:.1f} | Best: {best_reward:.1f}")
            elif ep == episodes-1:
                agent.save_models(save_path + "_final")
                print(f"Saved final model at episode 1000 | Reward: {episode_reward:.1f} | Best: {best_reward:.1f}")
            
            # Progress reporting
            if ep % 50 == 0 or ep == episodes-1:
                avg_battery = np.mean([uav['battery'] for uav in env.uavs])
                print(f"Ep {ep} | R:{episode_reward:.1f} | AvgR:{np.mean(env.metrics['episode_rewards'][-10:]):.1f} | ε:{agent.agents[0].epsilon:.3f} | Bat:{avg_battery:.1f}%")
        
        return agent
    
    except KeyboardInterrupt:
        print("\nTraining interrupted - saving models...")
        agent.save_models(save_path + "_interrupted")
        return agent
    finally:
        plt.close('all')

# ================== Modified Evaluation Function ==================
def evaluate(agent, env, n_episodes=20):
    test_rewards = []
    success_rates = []
    battery_levels = []
    charging_counts = []

    metrics = {
        'effective_charging': [],  # Charging when actually needed
        'energy_utilization': [],  # (Energy consumed)/(Energy charged)
        'priority_success': {0:[], 1:[], 2:[]}  # Success by task priority
    }
    
    for ep in range(n_episodes):
        states = env.reset()
        episode_reward = 0
        done = False
        charging_actions = 0

        while not done:
            actions = agent.act(states)
            charging_actions += sum(1 for a in actions if a == 4)
            next_states, rewards, dones, _ = env.step(actions)
            episode_reward += np.mean(rewards)
            states = next_states
            done = all(dones)

        test_rewards.append(episode_reward)
        success_rates.append(np.mean([1 if uav['current_task'] is None else 0 for uav in env.uavs]))
        battery_levels.append(np.mean([uav['battery'] for uav in env.uavs]))
        charging_counts.append(charging_actions/env.max_steps)

    if len(env.metrics['task_completion_rate']) > 0:
        ep_success = np.mean(env.metrics['task_completion_rate'][-env.max_steps:])
    else:
        ep_success = 0
        success_rates.append(ep_success)

    # Save evaluation plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.plot(test_rewards)
    plt.title("Test Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    plt.subplot(132)
    metrics = ['Success', 'Battery', 'Charging']
    values = [np.mean(success_rates)*100, np.mean(battery_levels), np.mean(charging_counts)*100]
    plt.bar(metrics, values)
    plt.title("Performance Metrics")
    plt.ylabel("Percentage (%)")
    
    plt.subplot(133)
    plt.plot(battery_levels)
    plt.title("Battery Levels")
    plt.xlabel("Episode")
    plt.ylabel("Battery (%)")
    
    plt.tight_layout()
    plt.savefig('final_evaluation.png')
    plt.close()

    # Print comprehensive results
    print("\nEnhanced Evaluation Results:")
    print(f"Average Reward: {np.mean(test_rewards):.1f} ± {np.std(test_rewards):.1f}")
    print(f"Task Success Rate: {np.mean(success_rates):.1%}")
    print(f"Final Battery Levels: {np.mean(battery_levels):.1f}%")
    print(f"Charging Frequency: {np.mean(charging_counts):.1%} of steps")
    print(f"Best Episode Reward: {max(test_rewards):.1f}")
    print(f"Consistency: {np.mean(test_rewards)/max(test_rewards):.1%} of best performance")
    print(f"Energy Balance: {sum(u['battery'] for u in env.uavs)/env.num_uavs:.1f}% "
          f"(Δ:{env.metrics.get('energy_consumed', [0])[-1] - env.metrics.get('energy_charged', [0])[-1]:.2f})")
    print(f"Charging Spread: {np.std([u['battery'] for u in env.uavs]):.2f} "
          f"(Min:{min(u['battery'] for u in env.uavs):.1f} "
          f"Max:{max(u['battery'] for u in env.uavs):.1f})")
    print(f"Charging Asymmetry: {abs(env.uavs[0]['battery'] - env.uavs[1]['battery']):.1f}% "
          f"(Thresholds: {env.uavs[0]['charging_profile']['min_threshold']}/{env.uavs[1]['charging_profile']['min_threshold']})")

if __name__ == "__main__":
    env = UAVIoVEnv()
    trained_agent = train_system(env, episodes=1000)
    evaluate(trained_agent, env, n_episodes=20)