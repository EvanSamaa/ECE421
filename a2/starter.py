import torch
from torch.utils.data import dataset, dataloader
import numpy as np
import matplotlib.pyplot as plt
import time
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(125)
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
    np.random.seed(324)
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
    s = s - (np.max(s, axis=1)).reshape(s.shape[0],1)

    # Softmax
    x = np.exp(s) / (np.sum(np.exp(s), axis=1)).reshape(s.shape[0],1)
    return x


def computeLayer(x, W, b):

    # Product of layer (Note activation function still needs to be applied)
    s = np.dot(x, W) + np.transpose(b)
    return s


def CE(target, prediction):

    # Cross Entropy loss for target and prediction
    loss = (-1 / target.shape[0]) * np.sum(target * np.log(prediction))
    return loss


def gradCE(y, s):

    # Find prediction
    x = softmax(s)

    return x - y

def evaluate_accuracy(y_hat, y):

    y_hat = torch.tensor(y_hat)
    y_hat = torch.argmax(y_hat, dim=1)
    y_hat = np.array(y_hat)
    accuracy = np.where(y_hat == y, 1, 0).sum()
    return(accuracy/y.shape[0])


class Cnn_model(torch.nn.Module):
    def __init__(self):
        super(Cnn_model, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, stride=1, kernel_size=3)
        self.relu = torch.nn.functional.relu
        self.batchNorm = torch.nn.BatchNorm2d(num_features=32)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(5408, 784)
        self.fc2 = torch.nn.Linear(784, 10)
        self.max = torch.nn.functional.softmax
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.batchNorm(x))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.max(self.fc2(x),dim=1)
        return x


class MyCustomDataset(dataset.Dataset):
    def __init__(self, data, label):
        data = torch.tensor(data).float()
        size = data.size()
        data = data.reshape((size[0], 1, size[1], size[2]))
        self.data = data
        self.label = torch.tensor(label).long()

    def __getitem__(self, index):
        # stuff
        return self.data[index, :, :], self.label[index]

    def __len__(self):
        return self.label.size()[0]

def train_torch_model(lr = 0.0001, epoch = 50):
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    # obtain dataloader
    trainDataLoader = dataloader.DataLoader(
        MyCustomDataset(trainData, trainTarget), batch_size=32
    )

    # set up model and optimizer
    cnn = Cnn_model()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {"params": cnn.conv1.weight},
        {"params": cnn.conv1.bias},
        {"params": cnn.fc1.weight, 'weight_decay':weight_decay},
        {"params": cnn.fc1.bias, 'weight_decay':weight_decay},
        {"params": cnn.fc2.weight, 'weight_decay': weight_decay},
        {"params": cnn.fc2.bias, 'weight_decay': weight_decay}
    ], lr=lr, weight_decay=0)
    del trainData, trainTarget # these are no longer used

    # training loop
    for i in range(0, epoch):

        # batches are achieved through the dataloader class
        for data, label in trainDataLoader:

            optimizer.zero_grad()
            cnn.train() # this is for batch normalization
            y_hat = cnn(data)
            loss = loss_func(y_hat, label)
            acc = evaluate_accuracy(y_hat, label)
            loss.backward()
            optimizer.step()
            # calculate and store loss and accuracy
            error_train.append(loss.item())
            acc_train.append(acc)
            optimizer.zero_grad()
            del loss, y_hat, acc, data, label
        cnn.eval() # this is for batch normalization

        # calculate loss and accuracy for testing and validation data
        with torch.no_grad():
            print(i)
            validation_output = cnn(change_shape_and_add_channel(validData))
            test_output = cnn(change_shape_and_add_channel(testData))

def forward(X, W_outer, b_outer, W_hidden, b_hidden):
    X = computeLayer(X, W_hidden, b_hidden)
    X = relu(X)
    X = computeLayer(X, W_outer, b_outer)
    return X

def train_numpy_model(hidden_dim, epochs=200):
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainTarget_one_hot, validTarget_one_hot, testTarget_one_hot = convertOneHot(trainTarget,validTarget,testTarget)
    trainData = trainData.reshape(-1, 28 * 28)
    validData = validData.reshape(-1, 28 * 28)
    testData = testData.reshape(-1, 28 * 28)

    d = trainData.shape[1]
    
    loss = [0] * epochs
    loss_valid = [0] * epochs
    loss_test = [0] * epochs
    accuracy = [0] * epochs
    accuracy_valid = [0] * epochs
    accuracy_test = [0] * epochs

    # Initialize Hyperparameters
    gamma = 0.9
    alpha = 0.00001

    # Initialize Weights
    d1 = trainData.shape[1]
    d2 = hidden_dim
    K = 10

    W_hidden = np.random.randn(d1, d2) * 2 / (d1 + d2)
    v_W_hidden = np.ones((d1, d2)) * 0.00001
    W_outer = np.random.randn(d2, K) * 2 / (d2 + K)
    v_W_outer = np.ones((d2, K)) * 0.00001
    b_hidden = np.random.randn(d2, 1) * 2 / (d2 + 1)
    v_b_hidden = np.ones((d2, 1)) * 0.00001
    b_outer = np.random.randn(K, 1) * 2 / (K + 1)
    v_b_outer = np.ones((K, 1)) * 0.00001

    for epoch in range(epochs):

        # Forward Step:
        s_0 = computeLayer(trainData, W_hidden, b_hidden)
        X_hidden = relu(s_0)
        s = computeLayer(X_hidden, W_outer, b_outer)

        s_valid = forward(validData, W_outer, b_outer, W_hidden, b_hidden)
        s_test = forward(testData, W_outer, b_outer, W_hidden, b_hidden)

        prediction = softmax(s)
        prediction_valid = softmax(s_valid)
        prediction_test = softmax(s_test)

        # Backward Step:
        grad_loss = gradCE(trainTarget_one_hot, s)
        grad_W_outer = np.dot(np.transpose(X_hidden), grad_loss)
        grad_b_outer = np.transpose(sum(grad_loss)).reshape(K, 1)
        grad_W_hidden = np.dot(np.transpose(trainData), np.where(X_hidden > 0, 1, 0) * np.dot(grad_loss, np.transpose(W_outer))) 
        grad_b_hidden = sum(np.where(X_hidden > 0, 1, 0) * np.dot(grad_loss, np.transpose(W_outer))).reshape(hidden_dim, 1)
        


        # Update Parameters
        v_W_outer = gamma * v_W_outer + alpha * grad_W_outer
        W_outer -= v_W_outer
        v_b_outer = gamma * v_b_outer + alpha * grad_b_outer
        b_outer -= v_b_outer
        v_W_hidden = gamma * v_W_hidden + alpha * grad_W_hidden
        W_hidden -= v_W_hidden
        v_b_hidden = gamma * v_b_hidden + alpha * grad_b_hidden
        b_hidden -= v_b_hidden

        loss[epoch] = CE(trainTarget_one_hot, prediction)
        loss_valid[epoch] = CE(validTarget_one_hot, prediction_valid)
        loss_test[epoch] = CE(testTarget_one_hot, prediction_test)

        accuracy[epoch] = evaluate_accuracy(prediction, trainTarget)
        accuracy_valid[epoch] = evaluate_accuracy(prediction_valid, validTarget)
        accuracy_test[epoch] = evaluate_accuracy(prediction_test, testTarget)

        print()
        print('EPOCH:', epoch)
        print('TRAINING:')
        print('    LOSS:', loss[epoch])
        print('    ACC:', accuracy[epoch])
        print('VALIDATION:')
        print('    LOSS:', loss_valid[epoch])
        print('    ACC:', accuracy_valid[epoch])
        print('TESTING:')
        print('    LOSS:', loss_test[epoch])
        print('    ACC:', accuracy_test[epoch])



    print()
    print('Final Training Accuracy:', round(accuracy[-1], 3))
    print('Final Validation Accuracy:', round(accuracy_valid[-1], 3))
    print('Final Testing Accuracy:', round(accuracy_test[-1], 3))

    # Plot Losses and Accuracy
    plot_epoch = [i for i in range(1, epochs + 1)]
    plt.figure()
    plt.plot(plot_epoch, loss, label='Training Loss')
    plt.plot(plot_epoch, loss_valid, label='Validation Loss')
    plt.plot(plot_epoch, loss_test, label='Testing Loss')
    plt.title('Losses as Function of Epoch, Hidden Size = '+str(hidden_dim))
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

    # Plot accuracy
    plt.figure()
    plt.plot(plot_epoch, accuracy, label='Training Accuracy')
    plt.plot(plot_epoch, accuracy_valid, label='Validation Accuracy')
    plt.plot(plot_epoch, accuracy_test, label='Testing Accuracy')
    plt.title('Accuracy as Function of Epoch, Hidden Size = '+str(hidden_dim))
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    hidden_size = 2000
    epoch = 200

    train_numpy_model(hidden_size)
    train_torch_model()



