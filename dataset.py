import torch
import numpy as np
from scipy import spatial

class Dataset(torch.utils.data.Dataset):
    def __init__(self, pts, knn, transform=None):
        tree = spatial.KDTree(pts)
        dists, indices = tree.query(pts, k=knn+1)
        radius = dists[:,-1]
        
        self.transform = transform
        self.pts = pts.astype(np.float32)
        self.radius = radius

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, idx):
        p = self.pts[idx]
        r = self.radius[idx]

        if self.transform:
            p = self.transform(p)

        return p, r