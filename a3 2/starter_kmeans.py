# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import torch

# Loading data
data = np.load('data2D.npy')
data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

# For Validation set
is_valid = True
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

def distanceLossFunction(X, MU):
  pair_dist = distance_func_torch(X, MU)
  min = pair_dist.min(dim = 1).values
  return min.sum()


# distance Function, but change into tensors first
def distance_func_torch(X, MU):
  pair_dist = torch.zeros((X.size()[0], MU.size()[0]))
  ones = torch.ones((X.size()[0], 1))
  for i in range(0, MU.size()[0]):
    col = X - torch.mm(ones, MU[i, :].reshape((MU[i, :].size()[0], 1)).T)
    pair_dist[:, i] = col.norm(dim = 1)
  return pair_dist


# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the squared pairwise distance matrix (NxK)
    # TODO
    pair_dist = np.zeros((X.shape[0], MU.shape[0]))
    ones = np.ones((X.shape[0], 1))
    for i in range(0, MU.shape[0]):
      # Calculate the difference between one center mu with all sample
      col = X - ones @ MU[i, :].reshape((MU[i, :].shape[0], 1)).T
      # find the norm
      pair_dist[:, i] = np.linalg.norm(col, axis = 1)
    # square the norm since the requirement calls for squared distance
    pair_dist = np.multiply(pair_dist, pair_dist)
    return pair_dist

# for plotting
def plot_losses(loss_list, label_list, save_name = "no_name.png"):
  for i in range(0, len(loss_list)):
    loss = loss_list[i]
    y = np.array(loss)
    x = np.arange(0, y.shape[0])
    plt.plot(x, y, label=label_list[i])
  plt.legend()
  plt.xlabel("Number of updates")
  plt.ylabel("Loss")
  plt.savefig(save_name)

def one_K_cluster(x_matrix, k = 3):

  data = x_matrix
  num_of_epochs = 100

  # to initialize MU
  avg = torch.zeros((k, data.shape[1]))
  std = torch.ones(k, data.shape[1])
  MU = torch.normal(avg, std)
  MU.requires_grad = True

  # change data to a tensor
  data = torch.tensor(data)

  # set up optimizer
  optimizer = torch.optim.Adam([{'params': MU, 'lr': 0.1, 'betas': (0.9, 0.99), 'eps': 1 * 10 ** -5}])

  losses = []
  for epoch in range(0, num_of_epochs):
    optimizer.zero_grad() # elimiate the gradient from last iteration
    loss = distanceLossFunction(data, MU) # calculate loss
    loss.backward() # backprop gradient
    optimizer.step() # update MU
    losses.append(loss.item())
  # plot_losses([losses], ["K = " + str(k)], save_name="1_1_loss")
  return MU



# return a (N, 1) array that shows which cluster k the data belongs to
def calculateOwnerShipPercentage(X, MU):
  k = MU.shape[0]
  pair_dist = distanceFunc(X, MU)
  ownership = np.argmin(pair_dist, axis=1)
  percentages = np.zeros((k, ))
  for item in ownership:
    percentages[item] = percentages[item] + 1
  percentages = percentages/percentages.sum()

  print(percentages)

  return ownership

# given x (N, D), the ownership returned from "calculateOwnerShipPercentage", mu (K,D) and the number of clusters k,
# this function plots the scatter graph
def plot_scatter(x, ownership, mu, k):

  ownership = ownership.reshape((ownership.shape[0], 1))
  x = np.concatenate((ownership, x), axis = 1)
  nan_arr = np.ones(x.shape) * np.nan
  mask = np.zeros(x.shape)
  mask[:, 0] = np.ones((x.shape[0], ))

  for i in range (0, k):
    data_plotted = np.where(np.multiply(x, mask) == mask*i, x, nan_arr)
    data_plotted = data_plotted[~np.isnan(data_plotted).any(axis=1)]
    data_plotted = data_plotted[:, 1:]
    x_vals = data_plotted[:, 0]
    y_vals = data_plotted[:, 1]
    data_label = "set " + str(i)
    plt.scatter(x_vals,y_vals, label=data_label)

  plt.scatter(mu[:, 0], mu[:, 1], label="data centers", c="#000000", marker="x")
  plt.legend()
  fileName = "1_2_k=" + str(k) + "scatter.png"
  plt.savefig(fileName)

if __name__ == "__main__":
  # clusters = []
  # for j in range (1, 6):
  #   clusters.append(one_K_cluster(data, k=j))
  val_loss = []
  for k in [5, 10, 15, 20, 30]:
  # for k in range (1, 6):
    plt.clf()
    plt.cla()
    mu = one_K_cluster(data, k).detach().numpy()
    loss = distanceLossFunction(torch.tensor(val_data), torch.tensor(mu))
    val_loss.append(loss)
    ownership = calculateOwnerShipPercentage(val_data, mu)
    plot_scatter(val_data, ownership, mu, k)
  
  print("validation loss", val_loss)