import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import torch

# Loading data
data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
 val_data = data[rnd_idx[:valid_batch]]
 data = data[rnd_idx[valid_batch:]]

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD Tensor (N observations and D dimensions)
    # MU: is an KxD Tensor (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK) Tensor
    z = torch.zeros((X.shape[0], MU.shape[0]))
    for i, x in enumerate(X):
        for j, mu in enumerate(MU):
            z[i,j] = torch.norm(x-mu)
    return z

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1
    # Outputs:
    # log Gaussian PDF N X K
    z=distanceFunc(X, mu)
    z=z*z/(sigma.T*torch.ones((X.shape[0],mu.shape[0])))
    log_gauss=-(z+torch.log(torch.Tensor(1/(2*np.pi*z))))/2
    return log_gauss


def log_posterior(log_PDF, log_pi):
    temp=torch.exp(log_PDF)
    temp=torch.sum(temp,1).unsqueeze(1)
    temp=torch.log(temp)
    return (log_PDF+log_pi.T)-temp.T
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

