# train.py
import argparse
import random, time, collections, os, pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import matplotlib.pyplot as plt

import gym
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from ddqn_model import DDQN

# ────────────────────────── Hyper‑parameters ──────────────────────────
BATCH_SIZE   = 64
GAMMA        = 0.9        # Discount factor as described in the paper
REPLAY_SIZE  = 100_000    # Experience replay buffer size
LEARNING_RATE = 0.0001    # Using RMSProp in the paper, but Adam works well too
TARGET_SYNC  = 10_000     # Steps between target network updates
START_EPS    = 1.0        # Starting epsilon for exploration
END_EPS      = 0.01       # Final epsilon value
EPS_DECAY_EP = 5000       # Episodes over which epsilon decays
MAX_EPISODES = 5000       # Total episodes for training as in the paper
STEPS_PER_EPISODE = 10000 # Max steps per episode
SAVE_EVERY   = 100        # Save weights every n episodes
EVAL_EVERY   = 50         # Evaluate the agent every n episodes
EVAL_EPISODES = 10        # Number of episodes for evaluation
LOG_DIR      = "logs"     # Directory to save logs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────── Utilities ─────────────────────────────
Transition = collections.namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done')
)

class ReplayMemory:
    """Experience replay buffer to store and sample transitions."""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    
    def __len__(self):
        return len(self.buffer)

def make_env():
    """Create and wrap the Super Mario Bros environment."""
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)    # (240,256,1)
    env = ResizeObservation(env, 84)                  # (84,84,1)
    try:
        env = FrameStack(env, 4, enable_lazy=True)    # (84,84,4) lazy frames
    except TypeError:
        env = FrameStack(env, 4)                      # Fallback for older gym versions
    return env

def obs_to_state(obs):
    """Convert LazyFrames (84,84,4) channel‑last → np.uint8 (4,84,84)."""
    state = np.array(obs)
    
    state = np.transpose(state, (3,0, 1, 2))
    # print(state.shape)
    return state   # Shape: (4, 84, 84)

def calculate_reward(info, prev_info=None):
    """
    Calculate reward based on game information.
    Combines score change and rightward movement as in the paper.
    """
    reward = 0
    
    # Base reward from the game (usually score changes)
    reward += info.get('reward', 0)
    
    # Reward for moving right (distance covered)
    if prev_info is not None:
        x_progress = info.get('x_pos', 0) - prev_info.get('x_pos', 0)
        reward += max(x_progress, 0) * 0.1  # Scale the reward for movement
    
    # Penalty for death
    if info.get('life', 0) < prev_info.get('life', 0) if prev_info else False:
        reward -= 1
        
    # Reward for completing the level
    if info.get('flag_get', False):
        reward += 50
        
    # Clip rewards to [-1, 1] range as mentioned in the paper
    return max(min(reward, 1), -1)

def evaluate_agent(env, agent, episodes=10, epsilon=0.01):
    """Evaluate the agent's performance over several episodes."""
    total_rewards = []
    
    for _ in range(episodes):
        state = env.reset()
        state = obs_to_state(state)
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < STEPS_PER_EPISODE:
            # ε-greedy policy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    # s = torch.from_numpy(state).unsqueeze(0).float().to(device)
                    q_values = agent(state)
                    action = q_values.argmax(1).item()
            
            next_state, reward, done, info = env.step(action)
            next_state = obs_to_state(next_state)
            total_reward += reward
            state = next_state
            steps += 1
        total_rewards.append(total_reward)
    
    return np.mean(total_rewards)

def plot_metrics(steps, rewards, q_values, epsilon_values, filename="training_metrics.png"):
    """Plot training metrics: rewards, average Q-values, and epsilon."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Plot rewards
    ax1.plot(steps, rewards)
    ax1.set_ylabel('Reward')
    ax1.set_title('Average Reward per Evaluation')
    
    # Plot Q-values
    ax2.plot(steps, q_values)
    ax2.set_ylabel('Average Q-value')
    ax2.set_title('Average Q-value during Training')
    
    # Plot epsilon values
    ax3.plot(steps, epsilon_values)
    ax3.set_ylabel('Epsilon')
    ax3.set_xlabel('Training Steps')
    ax3.set_title('Epsilon Decay')
    
    plt.tight_layout()
    plt.savefig(filename)

# ────────────────────────────── Training ──────────────────────────────
def train(path_policy=None, path_target=None):
    """Train a Double DQN agent to play Super Mario Bros."""
    # Set up logging directories
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Initialize metrics tracking
    eval_steps = []
    eval_rewards = []
    avg_q_values = []
    epsilon_values = []
    
    # Set up environment and models
    env = make_env()
    eval_env = make_env()  # Separate environment for evaluation
    n_actions = env.action_space.n
    
    # Initialize or load networks
    if path_policy and path_target and os.path.exists(path_policy) and os.path.exists(path_target):
        policy_net = DDQN(4, n_actions).to(device)
        target_net = DDQN(4, n_actions).to(device)
        policy_net.load_state_dict(torch.load(path_policy, map_location=device))
        target_net.load_state_dict(torch.load(path_target, map_location=device))
        print(f"Loaded pre-trained models from {path_policy} and {path_target}")
    else:
        policy_net = DDQN(4, n_actions).to(device)
        target_net = DDQN(4, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
    
    target_net.eval()  # Target network is only used for inference
    
    # Use RMSprop optimizer as mentioned in the paper
    optimizer = optim.RMSprop(
        policy_net.parameters(),
        lr=LEARNING_RATE,
        alpha=0.95,
        eps=0.01
    )
    
    memory = ReplayMemory(REPLAY_SIZE)
    
    # Training metrics
    all_scores = []
    
    # Track total steps across all episodes
    total_steps = 0
    
    # Progress bar for tracking episodes
    progress = trange(MAX_EPISODES, desc="Training", unit="episodes")
    
    # Main training loop over episodes
    for episode in progress:
        # Reset environment for new episode
        state = env.reset()
        state = obs_to_state(state)
        episode_reward = 0
        episode_q_values = []
        prev_info = None
        
        # Calculate epsilon based on episode number (linear decay)
        epsilon = max(END_EPS, START_EPS - (START_EPS - END_EPS) * episode / EPS_DECAY_EP)
        epsilon_values.append(epsilon)
        
        # Episode loop - maximum of 10,000 steps per episode as in the paper
        for step in range(STEPS_PER_EPISODE):
            total_steps += 1
            
            # Select action using ε-greedy policy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    # s = torch.from_numpy(state).unsqueeze(0).float().to(device)
                    q_values = policy_net(state)
                    action = q_values.argmax(1).item()
                    episode_q_values.append(q_values.mean().item())
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Calculate custom reward based on the paper
            modified_reward = calculate_reward(info, prev_info)
            prev_info = info.copy()
            
            # Process next state
            next_state = obs_to_state(next_state)
            
            # Store transition in replay memory
            memory.push(state, action, modified_reward, next_state, done)
            
            # Move to next state
            state = next_state
            episode_reward += reward  # Track original game reward for display
            
            # Update the Q-network at each step (when we have enough samples)
            if len(memory) >= BATCH_SIZE:
                # Sample a batch from replay memory
                transitions = memory.sample(BATCH_SIZE)
                
                # Convert batch to tensors
                batch_state = torch.from_numpy(np.stack(transitions.state)).float().to(device)
                batch_action = torch.tensor(transitions.action, dtype=torch.long, device=device).unsqueeze(1)
                batch_reward = torch.tensor(transitions.reward, dtype=torch.float32, device=device)
                batch_next_state = torch.from_numpy(np.stack(transitions.next_state)).float().to(device)
                batch_done = torch.tensor(transitions.done, dtype=torch.bool, device=device)
                
                # batch_action = batch_action.squeeze(1)
                batch_state = batch_state.squeeze(1)
                batch_next_state = batch_next_state.squeeze(1)
                # batch_reward = batch_reward.squeeze(1)
                # batch_done = batch_done.squeeze(1)
                
                # Compute current Q-values
                current_q_values = policy_net(batch_state).gather(1, batch_action).squeeze(1)
                
                # Compute next Q-values using Double DQN approach
                with torch.no_grad():
                    # Get actions from policy network
                    next_actions = policy_net(batch_next_state).max(1)[1].unsqueeze(1)
                    # Get Q-values from target network for those actions
                    next_q_values = target_net(batch_next_state).gather(1, next_actions).squeeze(1)
                    expected_q_values = batch_reward + GAMMA * next_q_values * (~batch_done)
                
                # Compute loss and optimize
                loss = nn.SmoothL1Loss()(current_q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients as mentioned in the paper
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()
            
            # Update target network periodically (every 10,000 steps as in the paper)
            if total_steps % TARGET_SYNC == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
            # Break if episode is done
            if done:
                break
        
        # End of episode processing
        all_scores.append(episode_reward)
        avg_score = np.mean(all_scores[-100:]) if len(all_scores) >= 100 else np.mean(all_scores)
        avg_q = np.mean(episode_q_values) if episode_q_values else 0
        
        # Update progress bar with episode metrics
        progress.set_postfix(
            episode=episode+1,
            score=f"{episode_reward:.1f}",
            avg_100=f"{avg_score:.1f}",
            avg_q=f"{avg_q:.3f}",
            eps=f"{epsilon:.3f}",
            steps=total_steps
        )
        
        # Periodically evaluate the agent and save model
        if (episode+1) % EVAL_EVERY == 0:
            eval_reward = evaluate_agent(eval_env, policy_net, episodes=EVAL_EPISODES)
            eval_steps.append(total_steps)
            eval_rewards.append(eval_reward)
            avg_q_values.append(avg_q)
            
            # Save the current model
            model_dir = pathlib.Path("models")
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(policy_net.state_dict(), model_dir / f"mario_ddqn_episode_{episode+1}.pth")
            
            # Plot and save metrics
            plot_metrics(
                eval_steps, 
                eval_rewards, 
                avg_q_values, 
                [max(END_EPS, START_EPS - (START_EPS - END_EPS) * e / EPS_DECAY_EP) for e in range(0, episode+1, EVAL_EVERY)],
                os.path.join(LOG_DIR, "training_metrics.png")
            )
    
    # Save final model
    torch.save(policy_net.state_dict(), "mario_ddqn_final.pth")
    env.close()
    eval_env.close()
    
    return policy_net
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Double DQN agent for Super Mario Bros.")
    parser.add_argument("--policy", type=str, help="Path to the policy network weights")
    parser.add_argument("--target", type=str, help="Path to the target network weights")
    parser.add_argument("--episodes", type=int, default=MAX_EPISODES, help="Number of training episodes")
    parser.add_argument("--steps_per_ep", type=int, default=STEPS_PER_EPISODE, help="Maximum steps per episode")
    
    args = parser.parse_args()
    MAX_EPISODES = args.episodes
    STEPS_PER_EPISODE = args.steps_per_ep
    
    # Create model directory if it doesn't exist
    model_dir = pathlib.Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the agent
    trained_agent = train(args.policy, args.target)