import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

class Buffed_LSTM(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, batch_first):
        super().__init__()
        self.BuffedHidden=None
        self.BuffedCell=None
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers, batch_first=batch_first)
        pass
    def forward(self, input):
        if self.BuffedHidden is None:
            output, (hidden, cell_state) = self.lstm(input)
            self.BuffedHidden=hidden
            self.BuffedCell=cell_state
        else:
            output, (hidden, cell_state) = self.lstm(input,(self.BuffedHidden, self.BuffedCell))
            self.BuffedHidden=hidden
            self.BuffedCell=cell_state
        return output

class DQN_LSTM(nn.Module):
    """mini CNN structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=8, kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(),
            # nn.LazyLinear(64),
            Buffed_LSTM(256,256,1,True),
            nn.ReLU(),
            nn.LazyLinear(output_dim)
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
        
if __name__ == '__main__':
    NET = DQN_LSTM((3,3,1), 3).float()
    print(NET.online.state_dict().keys())
    print(NET.target.state_dict().keys())
    print(NET.online.state_dict()['11.lstm.weight_hh_l0'])
    print(NET.target.state_dict()['11.lstm.weight_hh_l0'])
    pass