import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_dim):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 512)
        self.l4 = nn.Linear(512, 512 - input_dim)
        self.l5 = nn.Linear(512, 512)
        self.l6 = nn.Linear(512, 512)
        self.l7 = nn.Linear(512, 512)
        self.l_out = nn.Linear(512, 1)

    def forward(self, x):
        h = F.softplus(self.l1(x), beta=100)
        h = F.softplus(self.l2(h), beta=100)
        h = F.softplus(self.l3(h), beta=100)
        h = F.softplus(self.l4(h), beta=100)
        h = torch.cat((h, x), axis=1)
        h = F.softplus(self.l5(h), beta=100)
        h = F.softplus(self.l6(h), beta=100)
        h = F.softplus(self.l7(h), beta=100)
        h = self.l_out(h)
        return h
