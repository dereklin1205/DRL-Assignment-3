import torch
import torch.nn as nn
import torch.nn.functional as F

class DDQN(nn.Module):
    """
    Dueling Double DQN architecture with convolutional layers
    """
    def __init__(self, input_shape, n_actions):
        super(DDQN, self).__init__()
        
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate size of conv output
        conv_out_size = self._get_conv_out((input_shape, 84, 90))
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(torch.prod(torch.tensor(o.size())))
    
    def forward(self, x):
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
        batch_size = x.size(0)
        conv_out = self.conv(x).view(batch_size, -1)
        
        # Calculate value and advantage
        value = self.value_stream(conv_out)
        advantage = self.advantage_stream(conv_out)
        
        # Combine value and advantage streams
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + advantage - advantage.mean(dim=1, keepdim=True)