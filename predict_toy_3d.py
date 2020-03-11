import os
import random
import numpy as np
import colorsys
from skimage import measure

import open3d as o3d

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import build_network

def predict(net):
    x = np.linspace(-1.5, 1.5, 32)
    y = np.linspace(-1.5, 1.5, 32)
    z = np.linspace(-1.5, 1.5, 32)
    X, Y, Z = np.meshgrid(x, y, z)
    
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)
    pts = np.stack((X, Y, Z), axis=1)

    net.eval()
    val = net(torch.Tensor(pts))
    val = val.reshape(-1).detach().cpu().numpy()
    
    return pts, val

if __name__ == '__main__':
    net = build_network(input_dim=3)
    net.load_state_dict(torch.load('./models/toy_3d_model.pth', map_location='cpu'))
    
    pts, val = predict(net)
    volume = val.reshape(32, 32, 32)
    
    verts, faces, normals, values = measure.marching_cubes_lewiner(volume, 0.0)
    
    mesh = o3d.geometry.TriangleMesh()
    
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)

    os.makedirs('output', exist_ok=True)
    o3d.io.write_triangle_mesh("output/toy_3d_mesh.ply", mesh)
