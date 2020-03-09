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
    sampled = pts + torch.normal(0, 1, pts.shape) * noise.unsqueeze(1)
    return sampled

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

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    x = load_data('data/bunny/reconstruction/bun_zipper.ply')

    os.makedirs('output', exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    o3d.io.write_point_cloud("output/bunny_pts.ply", pcd)
   
    tree = spatial.KDTree(x)
    dists, indices = tree.query(x, k=51)
    radius = dists[:,-1]

    dataset = Dataset(x, radius)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    net = Network(input_dim=3)
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
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    os.makedirs('models', exist_ok=True)
    for itr in range(5000):
        loss = train(net, optimizer, data_loader, device)
        print(itr, loss)
        if itr % 100 == 0:
            torch.save(net.state_dict(), 'models/bunny_model_{0:04d}.pth'.format(itr))

    torch.save(net.state_dict(), 'models/bunny_model.pth')
