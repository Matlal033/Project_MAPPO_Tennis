import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Actor(nn.Module):

    def __init__(self, input_size, action_size, hidden_size, learing_rate, adam_epsilon):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
             nn.Linear(input_size, hidden_size),
             nn.ReLU(),
             nn.Linear(hidden_size, hidden_size),
             nn.ReLU(),
             nn.Linear(hidden_size, hidden_size),
             nn.ReLU(),
             nn.Linear(hidden_size, action_size)
        )

        self.std = nn.Parameter(torch.ones(1, action_size))
        self.optimizer = optim.Adam(self.parameters(), lr=learing_rate, eps=adam_epsilon)

    def forward(self, x):
        mu = 2*torch.tanh(self.actor(x)) #output limited to range ]-2,2[, plus std dev
        a_dist = torch.distributions.Normal(mu, self.std)

        return a_dist
    

class Critic(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, learing_rate, adam_epsilon):
        super(Critic, self).__init__()

        #input_size is qty of agents * size of observation an agent

        self.critic = nn.Sequential(
             nn.Linear(input_size, hidden_size),
             nn.ReLU(),
             nn.Linear(hidden_size, hidden_size),
             nn.ReLU(),
             nn.Linear(hidden_size, hidden_size),
             nn.ReLU(),
             nn.Linear(hidden_size, output_size)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learing_rate, eps=adam_epsilon)

    def forward(self, x):     
        values = self.critic(x).squeeze(-1)

        return values
