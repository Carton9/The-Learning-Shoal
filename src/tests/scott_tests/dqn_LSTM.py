import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

class Buffed_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)

    def forward(self, input, hidden_state=None, cell_state=None):
        if hidden_state is None or cell_state is None:
            output, (hidden, cell) = self.lstm(input)
        else:
            output, (hidden, cell) = self.lstm(input, (hidden_state, cell_state))
        return output, (hidden, cell)

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

    def forward(self, input, model, hidden_state=None, cell_state=None):
        if model == "online":
            # print(input.shape)
            features = self.online[:-3](input)
            features = features.unsqueeze(1)
            # print(features.shape)
            # if hidden_state is not None:
            #     print("hidden before:", hidden_state.shape)
            output, (hidden, cell) = self.online[-3](features, hidden_state, cell_state)
            # print("hidden:", hidden.shape) 
            output = F.relu(output)
            output = self.online[-1](output)
            output = output.squeeze(1)
            return output, (hidden, cell)
        elif model == "target":
            features = self.target[:-3](input)
            features = features.unsqueeze(1)
            output, (hidden, cell) = self.target[-3](features, hidden_state, cell_state)
            output = F.relu(output)
            output = self.target[-1](output)
            output = output.squeeze(1)
            return output, (hidden, cell)
        

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import copy

# class Buffed_LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, batch_first):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)

#     def forward(self, input, hidden_state=None, cell_state=None):
#         if hidden_state is None or cell_state is None:
#             output, (hidden, cell_state) = self.lstm(input)
#         else:
#             output, (hidden, cell_state) = self.lstm(input, (hidden_state.unsqueeze(1), cell_state.unsqueeze(1)))
#         return output, (hidden.squeeze(1), cell_state.squeeze(1))

# class DQN_LSTM(nn.Module):
#     """mini CNN structure
#   input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
#   """

#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         c, h, w = input_dim

#         self.online = nn.Sequential(
#             nn.Conv2d(in_channels=c, out_channels=8, kernel_size=3, stride=1,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.LazyLinear(256),
#             nn.ReLU(),
#             Buffed_LSTM(256,256,1,True),
#             nn.ReLU(),
#             nn.LazyLinear(output_dim)
#         )

#         self.target = copy.deepcopy(self.online)

#         # Q_target parameters are frozen.
#         for p in self.target.parameters():
#             p.requires_grad = False

#     def forward(self, input, model, hidden_state=None, cell_state=None):
#         if model == "online":
#             print(input.shape)
#             if hidden_state is not None:
#                 print(hidden_state.shape)
#                 print(cell_state.shape)

#             features = self.online[:-3](input)
#             print(features.shape)
            
#             output, (hidden, cell_state) = self.online[-3](features, hidden_state, cell_state)
#             output = F.relu(output)
#             output = self.online[-1](output)
#             return output, (hidden, cell_state)
#         elif model == "target":
#             features = self.target[:-3](input)
#             output, (hidden, cell_state) = self.target[-3](features, hidden_state, cell_state)
#             output = F.relu(output)
#             output = self.target[-1](output)
#             return output, (hidden, cell_state)
