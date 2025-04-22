# train.py
import argparse
import random, time, collections, os, pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gym
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from dqn_model import DQN

def preprocess_observation(observation):
    """Convert raw Mario observation to grayscale and resize to 84x84."""
    # Convert to grayscale
    gray = np.mean(observation, axis=2).astype(np.uint8)
    # Resize to 84x84
    from PIL import Image
    resized = np.array(Image.fromarray(gray).resize((84, 84), Image.BILINEAR))
    return resized

# Q-network
class Agent(object):
    """Agent that acts using a pre-trained DQN model."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(32, 12).to(self.device)  # Use 4 frames, not 32
        
        # Load pre-trained model
        self.policy_net.load_state_dict(torch.load("mario_dqn.pth", map_location=self.device))
        self.policy_net.eval()  # Set to evaluation mode
        
        # Initialize frame buffer with zeros for 4 frames
        self.frame_buffer = np.zeros((32, 84, 84), dtype=np.uint8)
    
    def act(self, observation):
        # Process the raw observation
        processed_frame = preprocess_observation(observation)
        
        # Update the frame buffer (oldest frame is replaced)
        self.frame_buffer = np.roll(self.frame_buffer, shift=-1, axis=0)
        self.frame_buffer[-1] = processed_frame
        
        # Convert stacked frames to tensor and get action
        stacked_frames = torch.tensor(self.frame_buffer, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(stacked_frames)
            action = q_values.max(1)[1].item()
        
        return action

# Do not modify the input of the 'act' function and the '__init__' function. 
if __name__ == "__main__":
    agent = Agent()
    # Setup the environment with the same wrappers used during training
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    
    observation = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.act(observation)
        print(action)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        
    print(f"Episode finished with total reward: {total_reward}")
    print(f"Final position: {info.get('x_pos', 0)}")
    env.close()