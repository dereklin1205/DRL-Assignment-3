# model.py
import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    Classic Atari‑style CNN from the DQN paper.
    Input  : (N, 4, 84, 84)  – four stacked grayscale frames
    Output : Q‑values for each discrete action
    """
    def __init__(self, in_ch: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=8, stride=2), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),    
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),    
            nn.ReLU()
        )
        # work out the linear layer size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(in_ch, 84, 84)
           ## self.conv(dummy) to one vector
            conv_out = self.conv(dummy).numel()
        self.head = nn.Sequential(
            nn.Linear(conv_out, 512), 
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        

    def forward(self, x):
                 # scale 0‑255 → 0‑1
        if x.dtype != torch.float32:
            x = x.float() / 255.0
            
        x = self.conv(x)
        
        ## compress x [ channel, width, height] into one vector and remains batch dimension
        # print(x.shape)
        # print(x.view(x.size(0), -1).shape)
        return self.head(x.view(x.size(0), -1))
