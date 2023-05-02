import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN_Basic(nn.Module):
    def __init__(self, observation_shape, n_actions):
        # observation shape: (H,W,C) needs to be reshaped to (C,H,W)
        # n_actions: int the size of actions
        super(DQN_Basic, self).__init__()
        self.n_actions = n_actions
        self.observation_shape = observation_shape
        # rearrange input shape to (C, H, W)
        self.convolve_and_rearrange = nn.Sequential(
            nn.Conv2d(observation_shape[-1], 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.flatten = nn.Flatten()
        self.layer1 = nn.LazyLinear(128)
        self.layer2 = nn.LazyLinear(self.n_actions)
        self.softmax = nn.Softmax()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # (1, H, W, C)
        x = self.convolve_and_rearrange(x.permute(0, 3, 1, 2))  # rearrange input shape to (C, H, W)
        x = self.flatten(x)
        # print(x.shape)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = self.softmax(x)
        return x