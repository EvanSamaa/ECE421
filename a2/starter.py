import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target

def relu(s):

    # ReLU
    x = np.maximum(0, s)
    return x


def softmax(s):

    # Subtract max element from input array to prevent exponential overflow
    s = s - np.max(s)

    # Softmax
    x = np.exp(s) / np.sum(np.exp(s))
    return x


def computeLayer(x, W, b):

    # Product of layer (Note activation function still needs to be applied)
    s = np.dot(W, x) + b
    return s


def CE(target, prediction):

    # Cross Entropy loss for target and prediction
    loss = (-1 / target.shape[0]) * np.sum(target * np.log(prediction))
    return loss


def gradCE(y, s):

    # Find prediction
    x = softmax(s)

    # Construct derivative matrix
    A = -np.outer(x, x) + np.diag(x)
    grad = -np.dot(A, y / x)

    return grad

if __name__ == "__main__":
    X = np.random.random((5,4))
    W = np.random.random((4,12))

    y = np.random.random((5,))
    y_hat = np.random.random((5,))
    print(CE(y, y_hat))
    b = 3
