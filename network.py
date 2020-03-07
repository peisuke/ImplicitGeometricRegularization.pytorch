i/2bmport torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_dim):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_dim, 128)
        self.l2 = nn.Linear(128, 128 - input_dim)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, 128)
        self.l5 = nn.Linear(128, 1)

    def forward(self, x):
        h = F.softplus(self.l1(x))
        h = F.softplus(self.l2(h))
        h = torch.cat((h, x), axis=1)
        h = F.softplus(self.l3(h))
        h = F.softplus(self.l4(h))
        h = self.l5(h)
        return h

class NetworkLarge(nn.Module):
    def __init__(self, input_dim):
        super(NetworkLarge, self).__init__()
        self.l1 = nn.Linear(input_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 512)
        self.l4 = nn.Linear(512, 512 - input_dim)
        self.l5 = nn.Linear(512, 512)
        self.l6 = nn.Linear(512, 512)
        self.l7 = nn.Linear(512, 512)
        self.l8 = nn.Linear(512, 1)

    def forward(self, x):
        h = F.softplus(self.l1(x))
        h = F.softplus(self.l2(h))
        h = F.softplus(self.l3(h))
        h = F.softplus(self.l4(h))
        h = torch.cat((h, x), axis=1)
        h = F.softplus(self.l5(h))
        h = F.softplus(self.l6(h))
        h = F.softplus(self.l7(h))
        h = self.l8(h)
        return h
