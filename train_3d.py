import os
import random
import numpy as np
import colorsys
from scipy import spatial

import open3d as o3d

import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataset import Dataset
from network import Network

def generate_data(nb_data=2048, noise=0.0):
    theta = 2 * np.random.rand(nb_data) * np.pi - np.pi
    p = np.random.rand(nb_data)
    phi = np.arcsin((2 * p) - 1)
    x = np.cos(phi) * np.cos(theta);
    y = np.cos(phi) * np.sin(theta);
    z = np.sin(phi);
    pts = np.stack((x, y, z), axis=1)

    return pts

def sample_fake(pts, noise=0.3):
    sampled = pts.detach().numpy()
    sampled += np.random.randn(*sampled.shape) * noise
    return sampled

def train(net, optimizer, data_loader, device):
    net.train()

    total_loss = 0
    total_count = 0
    for batch in data_loader:
        batchsize = batch.shape[0]
        
        net.zero_grad()
       
        fake = torch.Tensor(sample_fake(batch, 0.1))

        batch = batch.to(device)
        y = net(batch)
        loss_pts = (y ** 2).sum()
        
        xv = autograd.Variable(fake, requires_grad=True)
        xv = xv.to(device)
        f = net(xv)
        g = autograd.grad(outputs=f, inputs=xv,
                        grad_outputs=torch.ones(f.size()).to(device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        eikonal_term = ((g.norm(2, dim=1) - 1) ** 2).mean()
        
        loss = loss_pts + eikonal_term
        
        total_loss += loss.item()
        total_count += batchsize

        loss.backward()
        optimizer.step()

    total_loss /= total_count
    return total_loss
   
if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    x = generate_data(nb_data=2048, noise=0.01)
    dataset = Dataset(x)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    net = Network(input_dim=3)
    net.to(device)

    optimizer = optim.Adam(net.parameters())

    for itr in range(5000):
        loss = train(net, optimizer, data_loader, device)
        if itr % 100 == 0:
            print(loss)

    os.makedirs('models', exist_ok=True)
    torch.save(net.state_dict(), 'models/model.pth')
