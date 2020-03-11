import os
import random
import argparse
import numpy as np
import colorsys
from skimage import measure
import tqdm

import open3d as o3d

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import build_network

def predict(net, device, nb_grid):
    x = np.linspace(-1.5, 1.5, nb_grid)
    y = np.linspace(-1.5, 1.5, nb_grid)
    z = np.linspace(-1.5, 1.5, nb_grid)
    X, Y, Z = np.meshgrid(x, y, z)
    
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)
    pts = np.stack((X, Y, Z), axis=1)
    pts = pts.reshape(512, -1, 3)

    val = []
    net.eval()
    for p in tqdm.tqdm(pts):
        v = net(torch.Tensor(p).to(device))
        v = v.reshape(-1).detach().cpu().numpy()
        val.append(v)
    pts = pts.reshape((-1, 3))
    val = np.concatenate(val)
    return pts, val

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--name', '-n', type=str, default='output', help='output model name')

    args = parser.parse_args()
    name = args.name
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    net = build_network(input_dim=3)
    net.to(device)
    net.load_state_dict(torch.load('./models/{}_model.pth'.format(name), map_location=device))

    nb_grid = 128
    pts, val = predict(net, device, nb_grid)
    volume = val.reshape(nb_grid, nb_grid, nb_grid)
    
    verts, faces, normals, values = measure.marching_cubes_lewiner(volume, 0.0, spacing=(1.0, -1.0, 1.0), gradient_direction='ascent')
    
    mesh = o3d.geometry.TriangleMesh()
    
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)

    os.makedirs('output', exist_ok=True)
    o3d.io.write_triangle_mesh("output/{}_mesh.ply".format(name), mesh)
