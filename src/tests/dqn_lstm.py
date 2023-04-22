import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN_Basic(nn.Module):
    def __init__(self, observation_shape, n_actions):
        super(DQN_Basic, self).__init__()
        self.n_actions = n_actions
        self.observation_shape = observation_shape
        self.convolve_and_rearrange = nn.Sequential(
            nn.Conv2d(observation_shape[-1], 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size=32, hidden_size=128, num_layers=1, batch_first=True)
        self.layer1 = nn.Linear(128, 128)
        self.layer2 = nn.Linear(128, self.n_actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.convolve_and_rearrange(x)
        x = self.flatten(x)
        x = x.view(x.size(0), 1, -1)  # Reshape the tensor to (batch_size, 1, flattened_size) for LSTM input
        x, _ = self.lstm(x)
        x = x.squeeze(1)  # Remove the sequence dimension
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = self.softmax(x)
        return x