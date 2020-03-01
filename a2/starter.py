import torch
from torch.utils.data import dataset, dataloader
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from model import LinearModel

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


def evaluate_accuracy(y_hat, y):
    y_hat = torch.argmax(y_hat, dim=1)
    if torch.cuda.is_available():
        y_hat = y_hat.cpu()
        y = y.cpu()
    accuracy = np.where(y_hat == y, 1, 0).sum()
    return accuracy / y.size()[0]


class Cnn_model(torch.nn.Module):
    def __init__(self):
        super(Cnn_model, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=32, stride=1, kernel_size=3
        )
        torch.nn.init.xavier_uniform(self.conv1.weight)
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
        x = self.max(self.fc2(x), dim=1)
        return x


class MyCustomDataset(dataset.Dataset):
    def __init__(self, data, label):
        data = change_shape_and_add_channel(data)
        self.data = data
        self.label = torch.tensor(label).long()

    def __getitem__(self, index):
        # stuff
        return self.data[index, :, :], self.label[index]

    def __len__(self):
        return self.label.size()[0]

# helper function to turn (N, W, H) into (N, 1, W, H)
def change_shape_and_add_channel(data):
    data = torch.tensor(data).float()
    size = data.size()
    data = data.reshape((size[0], 1, size[1], size[2]))
    return data

def plot_trend(list_of_data, data_names=["Train", "Validation", "Test"], data_title="Accuracy", y_label="Accuracy"):

    for data, name in zip(list_of_data, data_names):
        x_axis = np.arange(len(data))
        plt.plot(x_axis, np.array(data), label=name)
    # plt.title(data_title)
    plt.xlabel("Epochs")
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(data_title + ".png")
    # plt.show()
    plt.clf()



def train_torch_model(lr = 0.0001, epoch = 50, weight_decay = 0.5):
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print("on GPU")
    else:
        print("on CPU")
    error_train = []
    acc_train = []
    error_valid = []
    acc_valid = []
    error_test = []
    acc_test = []
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

            if not torch.cuda.is_available():
                error_valid.append(loss_func(validation_output, torch.LongTensor(validTarget)).item())
                error_test.append(loss_func(test_output, torch.LongTensor(testTarget)).item())
                acc_valid.append(evaluate_accuracy(validation_output,torch.LongTensor(validTarget)))
                acc_test.append(evaluate_accuracy(test_output, torch.LongTensor(testTarget)))
            else:
                error_valid.append(loss_func(validation_output, torch.LongTensor(validTarget).cuda()).item())
                error_test.append(loss_func(test_output, torch.LongTensor(testTarget).cuda()).item())
                acc_valid.append(evaluate_accuracy(validation_output, torch.LongTensor(validTarget).cuda()))
                acc_test.append(evaluate_accuracy(test_output, torch.LongTensor(testTarget).cuda()))
            del validation_output, test_output

    print("The final Training Accuracy is ", acc_train[-1])
    print("The final Validation Accuracy is ", acc_valid[-1])
    print("The final Testing Accuracy is ", acc_test[-1])
    num = int(len(error_train) / len(error_valid))
    error_train = [error_train[i] for i in range(0, len(error_train), num)]
    acc_train = [acc_train[i] for i in range(0, len(acc_train), num)]
    plot_trend([error_train, error_valid, error_test],
               data_title="torch_loss_2-3-01", y_label="Loss")
    plot_trend([acc_train, acc_valid, acc_test], data_title="torch_accuracy_2-3-05")


if __name__ == "__main__":
    train_torch_model()

    d1 = 0
    d2 = [100,500,2000]
    for hidden_size in d2:
        m = linearModel(d1, hidden_size, 10, [relu, softmax], CE, gradCE)
        # training stuff

    y = np.random.random((5,))
    y_hat = np.random.random((5,))



