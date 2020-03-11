import random
import numpy as np

import torch
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataset import Dataset
from utils import sample_fake
from utils import build_network
from utils import train

def generate_data(nb_data=128, noise=0.0):
    t = 2 * np.random.rand(nb_data) * np.pi
    r = 1.0 + np.random.randn(nb_data) * noise
    pts = np.stack((r * np.cos(t), r * np.sin(t)), axis=1)
    return pts

def predict(centers, device, threshold=0.4):
    x = np.linspace(-1.5, 1.5, 40)
    y = np.linspace(-1.5, 1.5, 40)
    X, Y = np.meshgrid(x, y)
    
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    pts = np.stack((X, Y), axis=1)

    net.eval()
    val = net(torch.Tensor(pts).to(device))
    val = val.reshape(-1).detach().cpu().numpy()

    return pts, val
    
def plot_data(x, y, v):
    plt.figure(figsize=(12, 6))

    plt.subplot(1,2,1)
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.scatter(x[:,0], x[:,1])
 
    plt.subplot(1,2,2)
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.scatter(y[:,0], y[:,1], c=v, cmap=cm.seismic, vmin=-0.5, vmax=0.5)
    plt.show()

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    x = generate_data(nb_data=128, noise=0.01)

    dataset = Dataset(x, knn=10)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    net = build_network(input_dim=2)
    net.to(device)

    optimizer = optim.Adam(net.parameters())

    for itr in range(1000):
        loss = train(net, optimizer, data_loader, device)
        if itr % 100 == 0:
            print(loss)

    y, v = predict(x, device)
    
    plot_data(x, y, v)