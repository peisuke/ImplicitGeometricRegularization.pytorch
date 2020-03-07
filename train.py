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
from network import NetworkLarge as Network

def load_data(filename, noise=0.0):
    pcd = o3d.io.read_point_cloud(filename)
    pts = np.asarray(pcd.points)
    
    size = pts.max(axis=0) - pts.min(axis=0)
    pts = 2 * pts / size.max()
    pts -= (pts.max(axis=0) + pts.min(axis=0)) / 2
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

    x = load_data('data/bunny/reconstruction/bun_zipper.ply')

    os.makedirs('output', exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    o3d.io.write_point_cloud("output/bunny_pts.ply", pcd)
    
    dataset = Dataset(x)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    net = Network(input_dim=3)
    net.to(device)

    optimizer = optim.Adam(net.parameters())

    os.makedirs('models', exist_ok=True)
    for itr in range(5000):
        loss = train(net, optimizer, data_loader, device)
        print(loss)
        if itr % 100 == 0:
            torch.save(net.state_dict(), 'models/bunny_model_{0:04d}.pth'.format(itr))

    torch.save(net.state_dict(), 'models/bunny_model.pth')
