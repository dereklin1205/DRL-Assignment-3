import argparse
import random, time, collections, os, pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from collections import deque
import gym
import gym_super_mario_bros
from gym.wrappers import TimeLimit
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from torchvision import transforms as T
from ddqn_model import DDQN
from priority_buffer import PrioritizedReplayBuffer

# ────────────────────────── Hyper‑parameters ──────────────────────────
FRAME_SKIP   = 4
FRAME_STACK  = 4
BATCH_SIZE   = 32
GAMMA        = 0.95
REPLAY_SIZE  = 100_000
LEARNING_RATE= 0.0001
TARGET_SYNC  = 10000          # steps between target network updates
START_EPS    = 1.0
END_EPS      = 0.05
EPS_DECAY_FR = 1_000_000      # frames over which ε decays
MAX_FRAMES   = 10000000       # total env steps
EPSILON_DECAY_RATE = 0.999995
SAVE_EVERY   = 100            # save weights every … steps
PRIORITIZED_REPLAY_ALPHA = 0.6  # Alpha parameter for prioritized replay
PRIORITIZED_REPLAY_BETA = 0.4   # Initial beta parameter for importance sampling
PRIORITIZED_REPLAY_BETA_FRAMES = 1_000_000  # Frames over which beta increases to 1

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
            T.ToPILImage(), T.Grayscale(), T.Resize((84,90)), T.ToTensor()
        ])
        self.observation_space = gym.spaces.Box(0.0,1.0,shape=(1,84,90),dtype=np.float32)
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
    env = TimeLimit(env, max_episode_steps=5000)
    return env  

def obs_to_state(obs):
    """Convert LazyFrames (84,84,4) channel‑last → np.uint8 (4,84,84)."""
    state = np.array(obs)  # Convert LazyFrames to NumPy array
    state = torch.from_numpy(state).float() / 255.0  # Normalize to [0, 1]
    state = state.unsqueeze(0)  # Rearrange to (1, C, H, W)
    return state  # C,H,W

# ────────────────────────────── Training ──────────────────────────────
def train(path_policy=None, path_target=None):
    def reload_and_train(path_policy, path_target):
        policy_net = DDQN(FRAME_STACK, n_actions).to(device)
        target_net = DDQN(FRAME_STACK, n_actions).to(device)
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
        target_net.eval()
    else:
        policy_net = DDQN(FRAME_STACK, n_actions).to(device)
        target_net = DDQN(FRAME_STACK, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    # Use prioritized replay buffer instead of regular replay memory
    memory = PrioritizedReplayBuffer(
        REPLAY_SIZE, 
        alpha=PRIORITIZED_REPLAY_ALPHA,
        beta_start=PRIORITIZED_REPLAY_BETA,
        beta_frames=PRIORITIZED_REPLAY_BETA_FRAMES
    )
    
    state = env.reset()
    state = obs_to_state(state)

    episode_reward = 0
    all_scores = []
    progress = trange(5000, dynamic_ncols=True, desc="Training", unit="episode")
    eps = START_EPS
    frame = 0
    truncated = False
    pre_score = 0
    
    for episode in progress:
        done = False
        curr_frame = 0
        pre_score = 0
        score = 0
        truncated = False
        state = env.reset()
        state = obs_to_state(state)
        # print(episode)
        while not done and not truncated:
            frame += 1
            curr_frame += 1
            
            # ε-greedy action selection
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = policy_net(state)
                    action = q.argmax(1).item()
                    
            next_obs, reward, done, info = env.step(action)
            truncated = info.get("TimeLimit.truncated", False)
            
            next_state = obs_to_state(next_obs)
            score = info.get("score", 0)
            # Modified reward shaping
            reward = reward + (score - pre_score) / 5
            
            # Add experience to prioritized replay buffer
            memory.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Optimize after enough warm‑up
            if len(memory) >= BATCH_SIZE:
                # Sample batch with priorities
                batch_state, batch_action, batch_reward, batch_next, batch_done, indices, weights = memory.sample(BATCH_SIZE, device)
                
                # Current Q-values
                q_values = policy_net(batch_state).gather(1, batch_action).squeeze(1)
                
                # Double DQN: use policy net to select actions, target net to evaluate them
                with torch.no_grad():
                    # Get actions from policy network
                    next_actions = policy_net(batch_next).argmax(dim=1, keepdim=True)
                    # Evaluate those actions using target network
                    next_q = target_net(batch_next).gather(1, next_actions).squeeze(1)
                    # Compute target Q values
                    target_q = batch_reward + GAMMA * next_q * (~batch_done)
                
                # Compute TD errors for updating priorities
                td_errors = torch.abs(q_values - target_q).detach().cpu().numpy()
                
                # Update priorities in buffer
                memory.update_priorities(indices, td_errors)
                
                # Apply importance sampling weights to loss
                loss = (weights * nn.SmoothL1Loss(reduction='none')(q_values, target_q)).mean()
                
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
                optimizer.step()
            
            # Target network sync
            if frame % TARGET_SYNC == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
            # Epsilon decay
            eps = max(END_EPS, eps * EPSILON_DECAY_RATE)
            
            # env.render()
            pre_score = score
            
        # Episode bookkeeping
        if done or truncated:
            all_scores.append(episode_reward)
            progress.set_postfix(
                ep=episode,
                score=episode_reward,
                avg_100=np.mean(all_scores[-100:]) if len(all_scores) > 0 else 0,
                eps=f"{eps:.02f}"
            )
            state = env.reset()
            state = obs_to_state(state)
            episode_reward = 0
            
        # Autosave
        if (episode+1) % SAVE_EVERY == 0:
            torch.save(policy_net.state_dict(), "mario_ddqn.pth")

    # Final save
    torch.save(policy_net.state_dict(), "mario_ddqn.pth")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DDQN agent with prioritized replay on Super Mario Bros.")
    parser.add_argument("--policy", type=str, help="Path to the policy network weights.")
    parser.add_argument("--target", type=str, help="Path to the target network weights.")
    args = parser.parse_args()
    
    # Check if the paths are provided
    if args.policy and args.target:
        print(f"Policy network path: {args.policy}")
        print(f"Target network path: {args.target}")
    else:
        print("No pre-trained model provided. Training from scratch.")
    
    # Create the directory to save the model
    policy_path = args.policy if args.policy else None
    target_path = args.target if args.target else None
    train(policy_path, target_path)