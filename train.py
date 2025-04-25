# train.py
import argparse
import random, time, collections, os, pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import gym
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from dqn_model import DQN

# ────────────────────────── Hyper‑parameters ──────────────────────────
BATCH_SIZE   = 32
GAMMA        = 0.95
REPLAY_SIZE  = 100_000
LEARNING_RATE= 0.0001
TARGET_SYNC  = 1_000          # steps between target network updates
START_EPS    = 1.0
END_EPS      = 0.05
EPS_DECAY_FR = 1_000_000      # frames over which ε decays
MAX_FRAMES   = 10000000      # total env steps
epsilon_decay_rate = 0.9999995
SAVE_EVERY   = 100        # save weights every … steps
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ────────────────────────────── Utilities ─────────────────────────────
Transition = collections.namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done')
)

class ReplayMemory:
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
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)    # (240,256,1)
    env = ResizeObservation(env, 84)                  # (84,84,1)
    try:
        env = FrameStack(env, 4, enable_lazy=True)   # gym >= 0.26 / gymnasium
    except TypeError:
        env = FrameStack(env, 4)  
      # (84,84,4) lazy frames
    return env

def obs_to_state(obs):
    """Convert LazyFrames (84,84,4) channel‑last → np.uint8 (4,84,84)."""
    state = np.array(obs)  
    # print (state) # copy from LazyFrames
    # print(state.shape)
    # squeeze
    # print(state.shape)
    # print(np.transpose(state, (2,0,1,3)).shape)
    #(4,84,84,1)
    
    new_state = np.transpose(state, (3,0,1,2))
    return new_state   # C,H,W

# ────────────────────────────── Training ──────────────────────────────
def train(path_policy=None, path_target=None):
    def reload_and_train(path_policy, path_target):
        policy_net = DQN(4, n_actions).to(device)
        target_net = DQN(4, n_actions).to(device)
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
        policy_net  = DQN(4, n_actions).to(device)
        target_net  = DQN(4, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory    = ReplayMemory(REPLAY_SIZE)

    state= env.reset()
    state     = obs_to_state(state)

    episode_reward = 0
    episode        = 1
    all_scores     = []
    progress = trange(5000, dynamic_ncols=True, desc="Training", unit="episode")
    eps = START_EPS
    frame = 0
    for episode in progress:
        # ε‑greedy schedule
        done = 0
        curr_frame = 0
        
        while not done and curr_frame < 10000:
            frame += 1
            # print(frame)
            curr_frame += 1
            
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    # s = torch.from_numpy(state).to(device)
                    q = policy_net(state)
                    action = q.argmax(1).item()
            next_obs, reward, done, info = env.step(action)
            next_state = obs_to_state(next_obs)
            memory.push(state, action, reward, next_state, done)
            state  = next_state
            episode_reward += reward
        # Optimize after enough warm‑up
            if len(memory) >= BATCH_SIZE*100:
                transitions = memory.sample(BATCH_SIZE)
                batch_state   = torch.from_numpy(np.stack(transitions.state)).to(device)
                batch_action  = torch.tensor(transitions.action, dtype=torch.long, device=device).unsqueeze(1)
                batch_reward  = torch.tensor(transitions.reward, dtype=torch.float32, device=device)
                batch_next    = torch.from_numpy(np.stack(transitions.next_state)).to(device)
                batch_done    = torch.tensor(transitions.done, dtype=torch.bool, device=device)
                ## batch state squeeze the second dimension [32,1,4,84,84] to [32,4,84,84]
                batch_state   = batch_state.squeeze(1)
                batch_next    = batch_next.squeeze(1)
                q_values      = policy_net(batch_state).gather(1, batch_action).squeeze(1)
                with torch.no_grad():
                    next_q     = target_net(batch_next).max(1)[0]
                    target_q   = batch_reward + GAMMA * next_q * (~batch_done)
                loss = nn.SmoothL1Loss()(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print("1232")
            # Target network sync
            if frame % TARGET_SYNC == 0:
                target_net.load_state_dict(policy_net.state_dict())
            eps = max(END_EPS, eps * epsilon_decay_rate)
        # Episode bookkeeping
        if done:
            all_scores.append(episode_reward)
            progress.set_postfix(
                ep=episode,
                score=episode_reward,
                avg_100=np.mean(all_scores[-100:]),
                eps=f"{eps:.02f}"
            )
            episode += 1
            state= env.reset()
            state    = obs_to_state(state)
            episode_reward = 0
        # Autosave
        if (episode+1) % SAVE_EVERY == 0:
            torch.save(policy_net.state_dict(), "mario_dqn.pth")
        # Epsilon decay
        
    # progress = trange(MAX_FRAMES, dynamic_ncols=True, desc="Training", unit="frame")
    
    # for frame_idx in progress:
    #     # ε‑greedy schedule
    #     eps = END_EPS + (START_EPS - END_EPS) * np.exp(-1. * frame_idx / EPS_DECAY_FR)
    #     if random.random() < eps:
    #         action = env.action_space.sample()
    #     else:
    #         with torch.no_grad():
    #             s = torch.from_numpy(state).to(device)
    #             q = policy_net(s)
    #             action = q.argmax(1).item()
    #     next_obs, reward, done, info = env.step(action)
    #     next_state = obs_to_state(next_obs)
    #     memory.push(state, action, reward, next_state, done)

    #     state  = next_state
    #     episode_reward += reward

    #     # Optimize after enough warm‑up
    #     if len(memory) >= BATCH_SIZE:
    #         transitions = memory.sample(BATCH_SIZE)
    #         batch_state   = torch.from_numpy(np.stack(transitions.state)).to(device)
    #         batch_action  = torch.tensor(transitions.action, dtype=torch.long, device=device).unsqueeze(1)
    #         batch_reward  = torch.tensor(transitions.reward, dtype=torch.float32, device=device)
    #         batch_next    = torch.from_numpy(np.stack(transitions.next_state)).to(device)
    #         batch_done    = torch.tensor(transitions.done, dtype=torch.bool, device=device)
    #         ## batch state squeeze the second dimension [32,1,4,84,84] to [32,4,84,84]
    #         batch_state   = batch_state.squeeze(1)
    #         batch_next    = batch_next.squeeze(1)
    #         q_values      = policy_net(batch_state).gather(1, batch_action).squeeze(1)
            
    #         with torch.no_grad():
    #             next_q     = target_net(batch_next).max(1)[0]
    #             target_q   = batch_reward + GAMMA * next_q * (~batch_done)
            
    #         loss = nn.SmoothL1Loss()(q_values, target_q)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     # Target network sync
    #     if frame_idx % TARGET_SYNC == 0:
    #         target_net.load_state_dict(policy_net.state_dict())

    #     # Episode bookkeeping
    #     if done:
    #         all_scores.append(episode_reward)
    #         progress.set_postfix(
    #             ep=episode,
    #             score=episode_reward,
    #             avg_100=np.mean(all_scores[-100:]),
    #             eps=f"{eps:.02f}"
    #         )
    #         episode += 1
    #         state= env.reset()
    #         state    = obs_to_state(state)
    #         episode_reward = 0

    #     # Autosave
    #     if (frame_idx+1) % SAVE_EVERY == 0:
    #         torch.save(policy_net.state_dict(), "mario_dqn.pth")

    # Final save
    torch.save(policy_net.state_dict(), "mario_dqn.pth")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent on Super Mario Bros.")
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
    
