import torch
import numpy as np
import gym
from torchvision import transforms as T
from collections import deque
import random
class Agent:
    """Agent that acts using a pre-trained RainbowDQN model."""
    def __init__(self, model_path='mario_rainbow_dqn_frame_6800000.pth', frame_stack=4):
        self.device = torch.device("cpu")
        self.skip_frame = 4
        self.skip_count = 0
        self.action = None
        # Initialize the action space based on COMPLEX_MOVEMENT
        self.action_space = gym.spaces.Discrete(12)
        
        # Load the pre-trained Rainbow DQN model
        from train_rainbow import RainbowDQN  # Import your model class
        self.policy_net = RainbowDQN(input_channels=4, n_actions=self.action_space.n)
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.eval()
        
        # Frame processing
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.start = True
        # Image transformation pipeline
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 84)),
            T.ToTensor()
        ])
        
        # Initialize frames buffer with zeros
        

    def preprocess_frame(self, frame):
        """Convert raw frame to grayscale, resize, and normalize."""
        processed = self.transform(frame)
        return processed  # Shape: (1, 84, 84)
    
    def stack_frames(self, new_frame):
        """Add new frame to stack and return stacked frames."""
        self.frames.append(new_frame)
        return stacked  # Shape: (4, 84, 84)
    
    def act(self, observation, epsilon=0.05):
        """Determine action based on current observation."""
        # Preprocess the observation
        processed_frame = self.preprocess_frame(observation)
        
        # Add to frame stack
        if self.start:
            self.frames.clear()
            self.start = False
            for _ in range(self.frame_stack):
                self.frames.append(processed_frame)
        if self.skip_count != 0 :
            self.skip_count +=1
            if self.skip_count == 4:
                self.skip_count = 0
                # Stack frames
            return self.action
        else:
            self.skip_count +=1
        # Stack frames
            self.frames.append(processed_frame)
            ## pick the action based on these frames
            if random.random() < epsilon:
                action = self.action_space.sample()
            else:
                with torch.no_grad():
                    # Add batch dimension and send to device
                    stacked_frames = torch.stack(list(self.frames), dim=0).to(self.device)
                    ## permute
                    stacked_frames = np.transpose(stacked_frames, (1, 0, 2, 3))  # Shape: (4, 84, 84)
                    
                    q_values = self.policy_net(stacked_frames)
                    action = q_values.max(1)[1].item()
                self.action = action
            return action

    def reset(self):
        """Reset the frame stack when starting a new episode."""
        for i in range(len(self.frames)):
            self.frames[i] = torch.zeros((1, 84, 84))


# Usage example:
if __name__ == "__main__":
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    
    # Create environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    
    # Create agent
    agent = Agent()
    
    # Run episode
    observation = env.reset()
    agent.reset()  # Reset frame stack
    done = False
    total_reward = 0
    
    while not done:
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        print(reward)
        
    print(f"Episode finished with total reward: {total_reward}")
    print(f"Final position: {info.get('x_pos', 0)}")
    env.close()