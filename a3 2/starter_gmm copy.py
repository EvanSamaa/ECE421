import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import torch

# Loading data
data = np.load("data100D.npy")
data = np.load("data2D.npy")
[num_pts, dim] = np.shape(data)

logsoftmax = torch.nn.LogSoftmax(dim=0)

# For Validation set
is_valid = True
if is_valid:
    valid_batch = int(num_pts / 3.0)
    np.random.seed(45689)
    rnd_idx = np.arange(num_pts)
    np.random.shuffle(rnd_idx)
    val_data = data[rnd_idx[:valid_batch]]
    data = data[rnd_idx[valid_batch:]]

def softmax(s):

    # Subtract max element from input array to prevent exponential overflow
    s = s.numpy()
    s = s - (np.max(s, axis=1)).reshape(s.shape[0],1)

    # Softmax
    x = np.exp(s) / (np.sum(np.exp(s), axis=1)).reshape(s.shape[0],1)
    x = torch.tensor(x)
    return x

"""
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
            z[i, j] = np.linalg.norm(X - MU)
    return z
"""

def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the squared pairwise distance matrix (NxK)
    MU = torch.Tensor(MU)
    MU = MU.detach().numpy()

    pair_dist = np.zeros((X.shape[0], MU.shape[0]))
    ones = np.ones((X.shape[0], 1))
    for i in range(0, MU.shape[0]):
        # Calculate the difference between one center mu with all sample
        col = X - ones @ MU[i, :].reshape((MU[i, :].shape[0], 1)).T
        # find the norm
        pair_dist[:, i] = np.linalg.norm(col, axis = 1, ord = 1)
    pair_dist = np.multiply(pair_dist, pair_dist)
    return pair_dist


def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1
    # Outputs:
    # log Gaussian PDF N X K
    z = distanceFunc(X, mu)
    sigma = torch.Tensor(sigma)
    sigma = sigma.detach().numpy()
    log_gauss = np.add(-X.shape[1] / 2 * np.log(2 * np.pi) - 0.5 * np.log(np.prod(np.exp(sigma))), - 0.5 * np.multiply(z, (1 / np.exp(sigma)).T))
    return torch.Tensor(log_gauss)


def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K
    real_log_pi = logsoftmax(torch.exp(log_pi))
    post = torch.add(log_PDF, real_log_pi.T)
    # denominator = torch.sum(torch.exp(post), dim=1)
    # post = torch.add(post.T, - torch.log(denominator))
    # post = post.T
    # print(torch.sum(torch.exp(log_pi.T), dim=1))
    return post

def posterior_loss(X, mu, sigma, log_pi):
    """
    log_PDF = log_GaussPDF(X, mu, sigma)
    loss = torch.add(log_PDF, log_pi.T)
    loss = - torch.sum(loss)
    """
    log_PDF = log_GaussPDF(X, mu, sigma)
    log_post = log_posterior(log_PDF, log_pi)

    loss = torch.exp(log_post)
    loss = torch.sum(loss, dim=1)
    loss = torch.log(loss)
    loss = torch.sum(loss)
    loss = - loss
    return loss 

def one_K_cluster(x_matrix, k=3):

    data = x_matrix
    data = torch.Tensor(data)

    num_of_epochs = 100

    # to initialize MU
    mu_avg = torch.zeros((k, data.shape[1]))
    mu_std = torch.ones(k, data.shape[1])
    MU = torch.normal(mu_avg, mu_std)

    s_avg = torch.zeros((k, 1))
    s_std = torch.ones(k, 1)
    sigma = torch.normal(s_avg, s_std)
    sigma = torch.ones(k, 1)
    pi = torch.ones(k, 1)
    pi = pi / k
    log_pi = logsoftmax(pi)

    MU.requires_grad = True
    sigma.requires_grad = True
    log_pi.requires_grad = True

    # change data to a tensor
    data = torch.tensor(data)

    # set up optimizer
    optimizer = torch.optim.Adam([MU, sigma, log_pi], lr=0.1)

    losses = []
    for epoch in range(0, num_of_epochs):
        """
        pi = torch.exp(log_pi)
        pi = pi.detach()
        pi = softmax(pi)
        pi = torch.Tensor(pi)
        log_pi = torch.log(pi)
        log_pi.requires_grad = True
        """
        optimizer.zero_grad()  # eliminate the gradient from last iteration
        loss = posterior_loss(data, MU, sigma, log_pi)
        loss.backward()  # backprop gradient
        print(loss)
        optimizer.step()  # update
        losses.append(loss.item())
    # plot_losses([losses], ["K = " + str(k)], save_name="1_1_loss")
    
    return [MU, sigma, log_pi]


def calculateOwnerShipPercentage(X, MU, sigma, log_pi):
    k = MU.shape[0]
    gaussPDF = log_GaussPDF(X, MU, sigma)
    log_pi = torch.tensor(log_pi)
    log_post = log_posterior(gaussPDF, log_pi)
    ownership = np.argmin(-log_post, axis=1)
    percentages = np.zeros((k,))
    for item in ownership:
        percentages[item] = percentages[item] + 1
    percentages = percentages / percentages.sum()

    print("X", X)
    print("MU", MU)
    print("sigma", np.exp(sigma))
    print("pi", torch.exp(logsoftmax(torch.exp(log_pi))))
    print(percentages)
    return ownership


def plot_scatter(x, ownership, mu, k):

    ownership = ownership.reshape((ownership.shape[0], 1))
    x = np.concatenate((ownership, x), axis=1)
    nan_arr = np.ones(x.shape) * np.nan
    mask = np.zeros(x.shape)
    mask[:, 0] = np.ones((x.shape[0],))

    for i in range(0, k):
        data_plotted = np.where(np.multiply(x, mask) == mask * i, x, nan_arr)
        data_plotted = data_plotted[~np.isnan(data_plotted).any(axis=1)]
        data_plotted = data_plotted[:, 1:]
        x_vals = data_plotted[:, 0]
        y_vals = data_plotted[:, 1]
        data_label = "set " + str(i)
        plt.scatter(x_vals, y_vals, label=data_label)

    plt.scatter(mu[:, 0], mu[:, 1], label="data centers", c="#000000", marker="x")
    plt.legend()
    fileName = "2_2_k=" + str(k) + "scatter.png"
    plt.savefig(fileName)


if __name__ == "__main__":

    for k in range(1, 6):
        plt.clf()
        plt.cla()
        [mu, sigma, log_pi] = one_K_cluster(data, k)
        mu = mu.detach().numpy()
        sigma = sigma.detach().numpy()
        log_pi = log_pi.detach().numpy()
        ownership = calculateOwnerShipPercentage(val_data, mu, sigma, log_pi)
        plot_scatter(val_data, ownership, mu, k)
