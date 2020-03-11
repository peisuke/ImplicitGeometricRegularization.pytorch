import os
import random
import numpy as np

import open3d as o3d

import torch
import torch.optim as optim

from dataset import Dataset

from utils import sample_fake
from utils import build_network
from utils import train

def generate_data(nb_data=2048, noise=0.0):
    theta = 2 * np.random.rand(nb_data) * np.pi - np.pi
    p = np.random.rand(nb_data)
    phi = np.arcsin((2 * p) - 1)
    x = np.cos(phi) * np.cos(theta);
    y = np.cos(phi) * np.sin(theta);
    z = np.sin(phi);
    pts = np.stack((x, y, z), axis=1)

    return pts
   
if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    x = generate_data(nb_data=2048, noise=0.01)

    os.makedirs('output', exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    o3d.io.write_point_cloud("output/toy_3d_pts.ply", pcd)
    
    dataset = Dataset(x, knn=10)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    net = build_network(input_dim=3)
    net.to(device)

    optimizer = optim.Adam(net.parameters())

    for itr in range(300):
        loss = train(net, optimizer, data_loader, device)
        if itr % 100 == 0:
            print(loss)

    os.makedirs('models', exist_ok=True)
    torch.save(net.state_dict(), 'models/toy_3d_model.pth')
