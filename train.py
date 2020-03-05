import random
import numpy as np
from scipy import spatial

import torch
import torch.autograd as autograd
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataset import Dataset
from network import Network

def generate_data(nb_data=128, noise=0.0):
    t = 2 * np.random.rand(nb_data) * np.pi
    r = 1.0 + np.random.randn(nb_data) * noise
    pts = np.stack((r * np.cos(t), r * np.sin(t)), axis=1)
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
        #fake = torch.randn(batch.shape) * 4 - 2

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

def predict(centers, device, threshold=0.4):
    x = np.linspace(-1.5, 1.5, 40)
    y = np.linspace(-1.5, 1.5, 40)
    X, Y = np.meshgrid(x, y)
    
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    pts = np.stack((X, Y), axis=1)

    mat = spatial.distance_matrix(pts, centers)
    pts = pts[mat.min(axis=1) < threshold]
    
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
    dataset = Dataset(x)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    net = Network(input_dim=2)
    net.to(device)

    optimizer = optim.Adam(net.parameters())

    for itr in range(1000):
        loss = train(net, optimizer, data_loader, device)
        if itr % 100 == 0:
            print(loss)

    y, v = predict(x, device)
    
    plot_data(x, y, v)
