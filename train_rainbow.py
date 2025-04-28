import argparse
import random, time, collections, os, pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange, tqdm
from collections import deque
import gym
import gym_super_mario_bros
from gym.wrappers import TimeLimit
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from torchvision import transforms as T

# ────────────────────────── Hyper‑parameters ──────────────────────────
FRAME_SKIP   = 4
FRAME_STACK  = 4
BATCH_SIZE   = 32
GAMMA        = 0.9
REPLAY_SIZE  = 80000
LEARNING_RATE= 0.00025
TARGET_SYNC  = 10000     # Sync target network every N frames
START_EPS    = 0.95
END_EPS      = 0.1
EPS_DECAY_FR = 5000000   # Frame over which epsilon decays
MAX_FRAMES   = 10000000  # Total env steps
SAVE_EVERY   = 100000    # Save weights every ... frames
LOG_EVERY    = 100     # Log stats every ... frames
WARM_UP_SIZE = 10000
EPOCH_SIZE   = 50000     # Define frames per epoch
DEATH_PENALTY = -100
BACKWARD_PENALTY = 0
STAY_PENALTY = 0
start_ep = 1
NUMEP = 10000

# PER Parameters
ALPHA = 0.6         # Priority exponent
BETA_START = 0.4    # Initial beta for importance sampling
BETA_END = 1.0      # Final beta value
BETA_FRAMES = 5000000  # Frames over which to anneal beta

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ────────────────────────────── Utilities ─────────────────────────────
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip
    def step(self,action):
        total_reward, done = 0.0, False
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done: break
        return obs, total_reward, done, info
    
class GrayScaleResize(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.transform = T.Compose([
            T.ToPILImage(), T.Grayscale(), T.Resize((84,84)), T.ToTensor()
        ])
        self.observation_space = gym.spaces.Box(0.0,1.0,shape=(1,84,84),dtype=np.float32)
    def observation(self, obs):
        return self.transform(obs)
    
## Frame stack
class FrameStack(gym.Wrapper):
    def __init__(self, env, stack):
        super().__init__(env)
        self.stack = stack; 
        self.frames = deque(maxlen=stack)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0.0,1.0,shape=(shp[0]*stack,shp[1],shp[2]),dtype=np.float32)
    def reset(self):
        obs = self.env.reset()
        for _ in range(self.stack): self.frames.append(obs)
        return np.concatenate(self.frames,axis=0)
    def step(self,action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return np.concatenate(self.frames,axis=0), reward, done, info

def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, FRAME_SKIP)
    env = GrayScaleResize(env)
    env = FrameStack(env, FRAME_STACK)
    env = TimeLimit(env, max_episode_steps=3000)
    return env  

def obs_to_state(obs):
    """Convert observation to state for neural network input."""
    state = np.array(obs)  # Convert to NumPy array
    state = torch.from_numpy(state).float() / 255.0  # Normalize to [0, 1]
    state = state.unsqueeze(0)  # Add batch dimension
    return state  # Shape: (1, C, H, W)

# ─────────────────────────── Noisy Linear Layer ─────────────────────────
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        # Initialize mu weights
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        # Initialize sigma weights
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

# ─────────────────────────── Dueling Rainbow DQN Model ────────────────────────
class RainbowDQN(nn.Module):
    def __init__(self, input_channels, n_actions):
        super(RainbowDQN, self).__init__()
        
        # Feature extraction CNN
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate convolution output size
        conv_output_size = 7 * 7 * 64
        
        # Value stream - dueling architecture
        self.value_stream = nn.Sequential(
            NoisyLinear(conv_output_size, 512),
            nn.ReLU(),
            NoisyLinear(512, 1)
        )
        
        # Advantage stream - dueling architecture
        self.advantage_stream = nn.Sequential(
            NoisyLinear(conv_output_size, 512),
            nn.ReLU(),
            NoisyLinear(512, n_actions)
        )
    
    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage (dueling architecture)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    
    def reset_noise(self):
        """Reset noise for all noisy layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# ─────────────────────────── SumTree for PER ────────────────────────────
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        return self.tree[0]
    
    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        
        return idx, self.tree[idx], self.data[data_idx]

# ─────────────────────────── Prioritized Replay Buffer ────────────────────────────
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance-sampling exponent
        self.beta_increment = beta_increment
        self.epsilon = epsilon  # Small constant to avoid zero priority
        self.max_priority = 1.0  # Initial max priority
    
    def push(self, state, action, reward, next_state, done, error=None):
        # Use max priority for new experiences
        priority = self.max_priority if error is None else (abs(error) + self.epsilon) ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = []
        indices = []
        weights = np.zeros(batch_size, dtype=np.float32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        # Calculate segment size
        segment = self.tree.total() / batch_size
        
        # Increase beta over time (annealing beta)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Sample from each segment
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            
            indices.append(idx)
            batch.append(data)
            priorities[i] = priority
        
        # Calculate importance sampling weights
        # The smallest priority has the highest weight
        probabilities = priorities / self.tree.total()
        weights = np.power(self.tree.n_entries * probabilities, -self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Unpack batch of transitions
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.cat(states).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.cat(next_states).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
    
    def __len__(self):
        return self.tree.n_entries

# ────────────────────────────── Training ──────────────────────────────
def train(path_policy=None, path_target=None):
    def reload_and_train(path_policy, path_target):
        policy_net = RainbowDQN(FRAME_STACK, n_actions).to(device)
        target_net = RainbowDQN(FRAME_STACK, n_actions).to(device)
        if path_policy:
            # Load the pre-trained model for the policy network
            policy_net.load_state_dict(torch.load(path_policy, map_location=device))
            print(f"Loaded policy network weights from {path_policy}")
        if path_target:
            # Load the pre-trained model for the target network
            target_net.load_state_dict(torch.load(path_target, map_location=device))
            print(f"Loaded target network weights from {path_target}")
        return policy_net, target_net
    
    env = make_env()
    n_actions = env.action_space.n
    
    if args.policy and args.target:
        policy_net, target_net = reload_and_train(args.policy, args.target)
    else:
        policy_net = RainbowDQN(FRAME_STACK, n_actions).to(device)
        target_net = RainbowDQN(FRAME_STACK, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
    
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = PrioritizedReplayBuffer(REPLAY_SIZE, alpha=ALPHA, beta=BETA_START)
    
    # Initialize frame counter and tracking variables
    frame = 0
    episode = 0
    epoch = 0
    episode_reward = 0
    all_rewards = deque(maxlen=100)  # Store last 100 episode rewards
    moving_avg_reward = 0
    best_avg_reward = float('-inf')
    eps = START_EPS
    
    # Initialize state
    state = env.reset()
    state = obs_to_state(state)
    pre_score = 0
    prev_life = None
    prev_x = None
    
    # Initialize epoch stats
    epoch_frames = 0
    epoch_episodes = 0
    epoch_rewards = []
    epoch_losses = []
    
    # Initialize progress bar for frames
    progress = trange(start_ep, start_ep + NUMEP, dynamic_ncols=True, desc="Training", unit="Episodes")
    
    # Main training loop
    for episode in progress:
        # Reset environment
        state = env.reset()
        state = obs_to_state(state)
        done = False
        episode_reward = 0
        pre_score = 0
        prev_life = None
        prev_x = None
        ## reset episode noise
        policy_net.reset_noise()
        while not done:
            # Update beta value for importance sampling
            beta = min(BETA_END, BETA_START + frame * (BETA_END - BETA_START) / BETA_FRAMES)
            
            # Select action based on current state
            
                # Use policy network to select best action
            with torch.no_grad():
                q_values = policy_net(state.to(device))
                action = q_values.max(1)[1].item()
            
            # Epsilon decay
            eps = max(END_EPS, START_EPS - (START_EPS - END_EPS) * frame / EPS_DECAY_FR)
            
            # Execute action
            next_obs, reward, done, info = env.step(action)
            truncated = info.get("TimeLimit.truncated", False)
            
            # Process observation and reward
            next_state = obs_to_state(next_obs)
            score = info.get("score", 0)
            
            # Reward shaping
            pre_score = score
            
            # Apply custom penalties
            done_flag = done and not truncated
            cr = reward
            x_pos, life = info.get('x_pos'), info.get('life')
            
            if x_pos is not None:
                if prev_x is None: prev_x = x_pos
                dx = x_pos - prev_x
                cr += BACKWARD_PENALTY if dx < 0 else STAY_PENALTY if dx == 0 else 0
                prev_x = x_pos
                
            if prev_life is None: 
                prev_life = life
            elif life < prev_life: 
                cr += DEATH_PENALTY
                prev_life = life
            
            # Add to episode reward
            episode_reward += reward
            
            # Store transition in replay memory
            memory.push(state, action, cr, next_state, done_flag)
            
            # Move to next state
            state = next_state
            
            # Increment frame counter
            frame += 1
            epoch_frames += 1
            
            # Train if enough samples in memory
            if len(memory) >= WARM_UP_SIZE:
                # Sample batch with priorities
                batch_state, batch_action, batch_reward, batch_next_state, batch_done, indices, weights = memory.sample(BATCH_SIZE)
                
                # Compute current Q values
                q_values = policy_net(batch_state)
                state_action_values = q_values.gather(1, batch_action.unsqueeze(1)).squeeze(1)
                
                # Double DQN: Use online network to select actions
                with torch.no_grad():
                    # Select actions using policy network
                    next_actions = policy_net(batch_next_state).max(1)[1].unsqueeze(1)
                    # Evaluate actions using target network
                    next_q_target = target_net(batch_next_state)
                    next_state_values = next_q_target.gather(1, next_actions).squeeze(1)
                    # Zero out terminal states
                    next_state_values[batch_done] = 0.0
                    # Expected Q values
                    expected_values = batch_reward + GAMMA * next_state_values
                
                # Compute TD error
                td_error = (state_action_values - expected_values).detach()
                
                # Update priorities in replay buffer
                memory.update_priorities(indices, td_error.abs().cpu().numpy())
                
                # Compute weighted MSE loss
                loss = (weights * F.smooth_l1_loss(state_action_values, expected_values, reduction='none')).mean()
                epoch_losses.append(loss.item())
                
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()
                
                # Reset noise in noisy layers
                policy_net.reset_noise()
                
                # Target network synchronization
                if frame % TARGET_SYNC == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                    target_net.eval()
            
            # Periodic saving
            if frame % SAVE_EVERY == 0:
                save_path = f"mario_rainbow_dqn_frame_{frame}.pth"
                torch.save(policy_net.state_dict(), save_path)
                print(f"\nSaved model at frame {frame} to {save_path}")
                
                # Also save a copy as the latest model
                torch.save(policy_net.state_dict(), "mario_rainbow_dqn_latest.pth")
            
            # Periodic logging
            if frame % LOG_EVERY == 0:
                progress.set_postfix(
                    epoch=epoch,
                    frames=frame,
                    eps=f"{eps:.3f}",
                    episodes=episode,
                    avg_reward=f"{moving_avg_reward:.2f}",
                    best_avg=f"{best_avg_reward:.2f}"
                )
        
        # Episode end handling
        all_rewards.append(episode_reward)
        epoch_rewards.append(episode_reward)
        moving_avg_reward = np.mean(all_rewards)
        
        # Update best average reward
        if moving_avg_reward > best_avg_reward:
            best_avg_reward = moving_avg_reward
            # Save best model
            torch.save(policy_net.state_dict(), "mario_rainbow_dqn_best.pth")
        
        # Epoch completion check
        if epoch_frames >= EPOCH_SIZE:
            epoch += 1
            mean_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0
            mean_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
            print(f"\nEpoch {epoch} completed:")
            print(f"  Average reward: {mean_epoch_reward:.2f}")
            print(f"  Average loss: {mean_epoch_loss:.4f}")
            print(f"  Episodes: {epoch_episodes}")
            print(f"  Epsilon: {eps:.3f}")
            
            # Reset epoch stats
            epoch_frames = 0
            epoch_episodes = 0
            epoch_rewards = []
            epoch_losses = []
    
    # Final save
    torch.save(policy_net.state_dict(), "mario_rainbow_dqn_final.pth")
    env.close()
    print(f"Training completed. Final model saved to mario_rainbow_dqn_final.pth")
    print(f"Best average reward: {best_avg_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Rainbow DQN agent on Super Mario Bros.")
    parser.add_argument("--policy", type=str, help="Path to the policy network weights.")
    parser.add_argument("--target", type=str, help="Path to the target network weights.")
    args = parser.parse_args()
    
    # Check if the paths are provided
    if args.policy and args.target:
        print(f"Policy network path: {args.policy}")
        print(f"Target network path: {args.target}")
    else:
        print("No pre-trained model provided. Training from scratch.")
    
    # Run training
    policy_path = args.policy if args.policy else None
    target_path = args.target if args.target else None
    train(policy_path, target_path)