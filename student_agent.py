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
import cv2
from gym.spaces import Box
from dqn_model import DQN

# class ResizeObservation(ObservationWrapper):
#     r"""Downsample the image observation to a square image."""

#     def __init__(self, env, shape):
#         super().__init__(env)
#         if isinstance(shape, int):
#             shape = (shape, shape)
#         assert all(x > 0 for x in shape), shape

#         self.shape = tuple(shape)

#         obs_shape = self.shape + self.observation_space.shape[2:]
#         self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

#     def observation(self, observation):
#         import cv2

#         observation = cv2.resize(
#             observation, self.shape[::-1], interpolation=cv2.INTER_AREA
#         )
#         if observation.ndim == 2:
#             observation = np.expand_dims(observation, -1)
#         return observation
# class GrayScaleObservation(ObservationWrapper):
#     r"""Convert the image observation from RGB to gray scale."""

#     def __init__(self, env, keep_dim=False):
#         super().__init__(env)
#         self.keep_dim = keep_dim

#         assert (
#             len(env.observation_space.shape) == 3
#             and env.observation_space.shape[-1] == 3
#         )

#         obs_shape = self.observation_space.shape[:2]
#         if self.keep_dim:
#             self.observation_space = Box(
#                 low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8
#             )
#         else:
#             self.observation_space = Box(
#                 low=0, high=255, shape=obs_shape, dtype=np.uint8
#             )

#     def observation(self, observation):
#         import cv2

#         observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
#         if self.keep_dim:
#             observation = np.expand_dims(observation, -1)
#         return observation
## process the observation as above for gray scale
# def preprocess_observation_gray(observation):
#     observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
#     observation = np.expand_dims(observation, -1)
#     return observation
# def preprocess_observation_resize(observation):
#     observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
#     if observation.ndim == 2:
#         observation = np.expand_dims(observation, -1)
#     return observation
# def preprocess_observation(observation):
#     """Convert raw Mario observation to grayscale and resize to 84x84."""
#     # Convert to grayscale
#     new_observation = preprocess_observation_gray(observation)
#     # Resize to 84x84
#     new_observation = preprocess_observation_resize(new_observation)
#     # Convert to uint8
#     new_observation = np.array(new_observation, dtype=np.uint8)
#     # Normalize to 0-1 range
#     new_observation = new_observation.astype(np.float32) / 255.0
#     new_observation = np.transpose(new_observation, (2, 0, 1))  # Change to (C, H, W) format
#     # print(new_observation)
#     new_observation = new_observation.squeeze(0)  # Reshape to (32, 84, 84)
#     # new_observation.transpose((2, 0, 1))  # Change to (C, H, W) format
#     # print(new_observation.shape)
#     return new_observation
def preprocess_observation(observation):
    state = np.array(observation)  
    # print (state) # copy from LazyFrames
    # print(state.shape)
    # squeeze
    # print(state.shape)
    # print(np.transpose(state, (2,0,1,3)).shape)
    #(4,84,84,1)
    new_state = np.transpose(state, (3,0,1,2))
    return new_state   # C,H,W
# # Q-network
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
        self.frame_buffer = np.zeros((1,32, 84, 84), dtype=np.uint8)
    
    def act(self, observation):
        # Process the raw observation
        processed_frame = preprocess_observation(observation)
        
        # print(processed_frame.shape)
        # print(processed_frame.shape)
        # Update the frame buffer (oldest frame is replaced)
        # self.frame_buffer = np.roll(self.frame_buffer, shift=-1, axis=0)
        # self.frame_buffer[-1] = processed_frame
        # # print(self.frame_buffer.shape)
        # # Convert stacked frames to tensor and get action
        # # print(self.frame_buffer.shape)
        # stacked_frames = torch.tensor(self.frame_buffer, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # print(stacked_frames.shape)
        # print(stacked_frames.shape)
        epsilon = 0.05
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = self.action_space.sample()  # Random action
        else:
            with torch.no_grad():
                q_values = self.policy_net(processed_frame)
                action = q_values.max(1)[1].item()
                # print(q_values)
                # print(q_values)
        return action

# Do not modify the input of the 'act' function and the '__init__' function. 
if __name__ == "__main__":
    agent = Agent()
    # Setup the environment with the same wrappers used during training
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)  # (240, 256, 1)
    env = ResizeObservation(env, 84)                # (84, 84, 1)
    env = FrameStack(env, 32)     # (84, 84, 4)
    observation = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.act(observation)
        print(action)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        # print(action)
        env.render()
        
    print(f"Episode finished with total reward: {total_reward}")
    print(f"Final position: {info.get('x_pos', 0)}")
    env.close()
    
    
    import numpy as np
# from gym.spaces import Box
# from gym import ObservationWrapper


# class ResizeObservation(ObservationWrapper):
#     r"""Downsample the image observation to a square image."""

#     def __init__(self, env, shape):
#         super().__init__(env)
#         if isinstance(shape, int):
#             shape = (shape, shape)
#         assert all(x > 0 for x in shape), shape

#         self.shape = tuple(shape)

#         obs_shape = self.shape + self.observation_space.shape[2:]
#         self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

#     def observation(self, observation):
#         import cv2

#         observation = cv2.resize(
#             observation, self.shape[::-1], interpolation=cv2.INTER_AREA
#         )
#         if observation.ndim == 2:
#             observation = np.expand_dims(observation, -1)
#         return observation
