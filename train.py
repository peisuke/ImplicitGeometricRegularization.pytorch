import random
import numpy as np

import torch
import torch.autograd as autograd
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataset import Dataset
from network import Network

def generate_data(nb_data=128):
    t = 2 * np.random.rand(nb_data) * np.pi
    pts = np.stack((np.cos(t), np.sin(t)), axis=1)
    return pts

def train(net, optimizer, data_loader):
    net.train()

    total_loss = 0
    total_count = 0
    for batch in data_loader:
        batchsize = batch.shape[0]
        
        net.zero_grad()

        y = net(batch)
        loss_pts = (y ** 2).sum()
        
        fake = torch.randn(batch.shape) * 4 - 2
        xv = autograd.Variable(fake, requires_grad=True)
        f = net(xv)
        g = autograd.grad(outputs=f, inputs=xv,
                        grad_outputs=torch.ones(f.size()),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        eikonal_term = ((g.norm(2, dim=1) - 1) ** 2).mean()
        
        loss = loss_pts + eikonal_term
        
        total_loss += loss.item()
        total_count += batchsize

        loss.backward()
        optimizer.step()

    total_loss /= total_count
    return total_loss

def predict():
    x = np.linspace(-1.5, 1.5, 30)
    y = np.linspace(-1.5, 1.5, 30)
    X, Y = np.meshgrid(x, y)
    
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    pts = np.stack((X, Y), axis=1)
    
    net.eval()
    val = net(torch.Tensor(pts))
    val = val.reshape(-1).detach().numpy()

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
    plt.scatter(y[:,0], y[:,1], c=v, cmap=cm.seismic)
    plt.show()

if __name__ == '__main__':
    x = generate_data(nb_data=128)
    dataset = Dataset(x)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    net = Network()
    optimizer = optim.Adam(net.parameters())

    for itr in range(1000):
        loss = train(net, optimizer, data_loader)
        if itr % 100 == 0:
            print(loss)

    y, v = predict()
    
    plot_data(x, y, v)
