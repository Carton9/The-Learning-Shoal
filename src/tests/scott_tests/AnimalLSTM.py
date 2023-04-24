from dqn_net import DQN_NET
import torch
import numpy as np
from torch import nn
from collections import deque
import random, datetime, os, copy
from dqn_LSTM import DQN_LSTM
class Animal:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = DQN_LSTM(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e4  # no. of experiences between saving Mario Net
        
        # cache and memory
        self.memory = deque(maxlen=50000)
        self.batch_size = 32

        # td estimate and td target
        self.gamma = 0.9

        # updating the model
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    def act(self, state, hidden, cell, active=True):
        """
    Given a state, choose an epsilon-greedy action and update value of step.
    """
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        if hidden != None and cell != None:
            hidden = torch.tensor(hidden, device=self.device).unsqueeze(0)
            cell = torch.tensor(cell, device=self.device).unsqueeze(0)
        else:
            hidden = None
            cell = None
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            _, (hidden, cell) = self.net(state, model="online", hidden_state=hidden, cell_state=cell)
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            action_values, (hidden, cell) = self.net(state, model="online", hidden_state=hidden, cell_state=cell)
            action_idx = torch.argmax(action_values, axis=1).item()

        if active:
            # decrease exploration_rate
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

            # increment step
            self.curr_step += 1
        return action_idx, (hidden, cell)
    
    
    def cache(self, state, next_state, hidden, cell, action, reward, done, agentName=None):
        """
        Store the experience to self.memory (replay buffer)
        """
        if hidden != None:
            hidden = np.array(torch.Tensor.cpu(hidden.detach()))
        if cell != None:
            cell = np.array(torch.Tensor.cpu(cell.detach()))
        # cell.to(device='cpu')
        print(type(state))
        print(type(next_state))
        print(type(hidden))
        print(type(cell))
        self.memory.append((state, next_state, hidden, np.array(cell), action, reward, done))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, hidden, cell, action, reward, done = map(np.stack, zip(*batch))
        
        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        hidden = torch.tensor(hidden, device=self.device)
        cell = torch.tensor(cell, device=self.device)
        action = torch.tensor(action, device=self.device, dtype=torch.int64).squeeze()
        reward = torch.tensor(reward, device=self.device).squeeze()
        done = torch.tensor(done, device=self.device).squeeze()

        return state, next_state, hidden, cell, action, reward, done
    
    def td_estimate(self, state, hidden, cell, action):
        current_Q, (current_Hidden, current_Cell) = self.net(state, hidden, cell, model="online")
        current_Q = current_Q[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        ##########################################
        # do something with hidden and cell states if needed

        ##########################################
        return current_Q, (current_Hidden, current_Cell)

    @torch.no_grad()
    def td_target(self, reward, next_state, hidden, cell, done):
        next_state_Q, _ = self.net(next_state, hidden, cell, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q, _ = self.net(next_state, hidden, cell, model="target")
        next_Q = next_Q[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())
    
    def save(self):
        save_path = (
            self.save_dir / f"animal_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"Animal saved to {save_path} at step {self.curr_step}")
    
    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, hidden, cell, action, reward, done = self.recall()

        # Get TD Estimate
        td_est, (current_hidden, current_cell) = self.td_estimate(state, hidden, cell, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, current_hidden, current_cell, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)