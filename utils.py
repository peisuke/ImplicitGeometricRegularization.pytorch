import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn

from network import Network

def sample_fake(pts, noise=0.3):
    sampled = pts + torch.normal(0, 1, pts.shape) * noise.unsqueeze(1)
    return sampled

def build_network(input_dim=3):
    net = Network(input_dim=input_dim)
    for k, v in net.named_parameters():
        if 'weight' in k:
            std = np.sqrt(2) / np.sqrt(v.shape[0])
            nn.init.normal_(v, 0.0, std)
        if 'bias' in k:
            nn.init.constant_(v, 0)
        if k == 'l_out.weight':
            std = np.sqrt(np.pi) / np.sqrt(v.shape[1])
            nn.init.constant_(v, std)
        if k == 'l_out.bias':
            nn.init.constant_(v, -1)            
    return net

def train(net, optimizer, data_loader, device):
    net.train()

    total_loss = 0
    total_count = 0
    for batch in data_loader:
        pts = batch[0]
        rad = batch[1]
        batchsize = pts.shape[0]
        
        net.zero_grad()

        fake = torch.Tensor(sample_fake(pts, rad)) 
        uniform = 3 * torch.rand_like(fake) - 1.5
        fake = torch.cat((fake, uniform), axis=0)

        pts = pts.to(device)
        y = net(pts)
        loss_pts = (y ** 2).sum()
        
        xv = autograd.Variable(fake, requires_grad=True)
        xv = xv.to(device)
        f = net(xv)
        g = autograd.grad(outputs=f, inputs=xv,
                        grad_outputs=torch.ones(f.size()).to(device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        eikonal_term = ((g.norm(2, dim=1) - 1) ** 2).mean()
        
        loss = loss_pts + 0.1 * eikonal_term
        
        total_loss += loss.item()
        total_count += batchsize

        loss.backward()
        optimizer.step()

    total_loss /= total_count
    return total_loss