import argparse
import os
import numpy as np

import open3d as o3d

import torch
import torch.optim as optim

from dataset import Dataset

from utils import sample_fake
from utils import build_network
from utils import train

def load_data(filename, noise=0.0):
    pcd = o3d.io.read_point_cloud(filename)
    pts = np.asarray(pcd.points)
    
    size = pts.max(axis=0) - pts.min(axis=0)
    pts = 2 * pts / size.max()
    pts -= (pts.max(axis=0) + pts.min(axis=0)) / 2
    return pts


def get_batchsize(iter):
    scheduler = [
        {'epoch': 10, 'batch_size': 32}, 
        {'epoch': 20, 'batch_size': 64}, 
        {'epoch': 30, 'batch_size': 128}, 
        {'epoch': 40, 'batch_size': 256}, 
        {'epoch': 50, 'batch_size': 512}, 
        {'epoch': 100, 'batch_size': 1024}
    ]
    for s in scheduler:
        if iter < s['epoch']:
            return s['batch_size']
    return 2048

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input', '-i', type=str, required=True, help='input filename (pcd, ply)')
    parser.add_argument('--name', '-n', type=str, default='output', help='output model name')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='output model name')
    parser.add_argument('--fast', action='store_true', help='batch size scheduling')

    args = parser.parse_args()
    input_path = args.input
    output_name = args.name
    nb_epochs = args.epochs
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    x = load_data(input_path)
    
    os.makedirs('output', exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    o3d.io.write_point_cloud("output/{}_pts.ply".format(output_name), pcd)
    
    dataset = Dataset(x, knn=50)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    net = build_network(input_dim=3)
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    os.makedirs('models', exist_ok=True)
    for itr in range(nb_epochs):
        if args.fast:
            batch_size = get_batchsize(itr)
            if batch_size != data_loader.batch_size:
                data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss = train(net, optimizer, data_loader, device)
        print(itr, loss)
        if itr % 100 == 0:
            torch.save(net.state_dict(), 'models/model_{0:04d}.pth'.format(itr))

    torch.save(net.state_dict(), 'models/{}_model.pth'.format(output_name))
