import tensorflow as tf
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import torch
import time
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        'Initialization'
        self.labels = labels
        self.data = data.reshape((data.shape[0], data.shape[1]*data.shape[1]))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)
    def __getitem__(self, index):
        feature = self.data[index]
        label = self.labels[index]
        return feature, label

class Prceptron(torch.nn.Module):
    def __init__(self, input_size, neuronCount=1, weight=None):
        super(Prceptron, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, neuronCount, bias=True)

    def forward(self, features):
        x = self.fc1(features)
        return x
def truncatedNormal(input_tensor, mean, std):
    for i in range (0, input_tensor.size()[0]):
        for j in range (0, input_tensor.size()[1]):
            num = np.random.normal(mean, std)
            while np.abs(num - mean) > 2 * std:
                num = np.random.normal(mean, std)
            input_tensor[i][j] = num
    return input_tensor.numpy()

def loadData():
    with np.load('notMNIST.npz') as data:
        Data, Target = data['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx] / 255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target == posClass] = 1
        Target[Target == negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def MSE(W, b, x, y, reg):
    try:
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[1])
    except:
        print("nothing")
    ones = np.ones((x.shape[0], 1)).squeeze()
    y_est = (x.dot(W) + b * ones).squeeze()
    y = y.squeeze()
    error = ((y - y_est).dot((y - y_est)) + 1 * reg / 2 * (W.squeeze().dot(W.squeeze()))) / x.shape[0]
    return error

def gradMSE(W, b, x, y, reg):
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[1])  # flattens the second and third dimensions
    ones = np.ones((x.shape[0], 1)).squeeze()
    y = y.squeeze()
    b_grad = (2 * ones.dot(x.dot(W) + b * ones - y)) / x.shape[0]
    W_grad = (2 * x.T.dot((x.dot(W) + b * ones - y)) + reg * W) / x.shape[0]
    return b_grad, W_grad

def CrossEntropyLoss(w, b, x, y, reg):
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[1])
    y = y.squeeze()
    y = y.astype(int)
    y = y*2-1
    N = x.shape[0]
    d = x.shape[1]
    I = np.ones(N,)
    #print(w.shape,np.array([b]).shape)
    y_est = x.dot(w) + b * I#loss = np.average(y*np.log(1+np.exp(-y_est))+(1-y)*np.log(1+np.exp(y_est))) + reg / 2 * np.linalg.norm(w)
    loss = np.average(np.log(1+np.exp(-y*y_est)))+reg*np.linalg.norm(np.concatenate((w.reshape(-1,1),np.array([b]).reshape(-1,1))))
    return loss

def gradCE(w, b, x, y, reg):
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[1])
    y = y.squeeze()
    y = y.astype(int)
    N = x.shape[0]
    d = x.shape[1]
    I = np.ones(N,)
    y_est = x.dot(w) + b * I
    y=y*2-1
    grad = -y / (1 + np.exp(y_est * y))
    grad = np.array([grad]).T
    w_grad = grad * x
    w_grad = np.average(w_grad,0) + reg*w
    b_grad = grad
    b_grad = np.average(b_grad,0) + reg*b
    return b_grad,w_grad

def compute_loss_with_tensor(model, data_set, loss_func):
    # temp_loader = torch.utils.data.DataLoader(dataSet, batch_size=len(dataSet), shuffle=True)
    data = data_set.data
    batch_label = data_set.labels
    result = model(data)
    loss = loss_func(result.squeeze(), batch_label.squeeze()).item()
    # print(loss)
    return loss
def evaluate(model, data_set):
    data = data_set.data
    batch_label = data_set.labels.numpy()
    result = model(data).detach().numpy()
    result = np.where(result >= 0.5, 1, 0).squeeze()
    accuracy = np.where(result == batch_label.squeeze(), 1, 0).sum()
    # print(accuracy/result.shape[0])
    return accuracy/result.shape[0]

def compute_accuracy(W, b, x, y):
    try:
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[1])
    except:
        pass
    output = x.dot(W) + b
    output = np.where(output >= 0.5, 1, 0).squeeze()
    accuracy = np.where(output == y.squeeze(), 1, 0).sum()
    return accuracy / output.shape[0]

def plot_trend(list_of_data, data_names=["Train", "Validation", "Test"], data_title="Accuracy", y_label="Accuracy"):
    x_axis = np.arange(len(list_of_data[0]))
    for data, name in zip(list_of_data, data_names):
        plt.plot(x_axis, np.array(data), label=name)
    # plt.title(data_title)
    plt.xlabel("Epochs")
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(data_title + ".png")
    plt.show()

def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol=0.0000001, val_data=[], test_data=[], lossType='MSE'):
    # initialize storage elements
    error_train = []
    acc_train = []
    error_valid = []
    acc_valid = []
    error_test = []
    acc_test = []
    lossTypeDic={'MSE':gradMSE, 'CE':gradCE}
    errorTypeDic={'MSE':MSE, 'CE':CrossEntropyLoss}
    loss_func=lossTypeDic[lossType]
    error_func=errorTypeDic[lossType]
    for epoch in range(0, epochs):
        # compute gradient
        b_grad, W_grad = loss_func(W, b, x, y, reg)
        # compute training error and accuracy
        accuracy = compute_accuracy(W, b, x, y)
        epoch_error = (error_func(W, b, x, y, reg))
        error_train.append(epoch_error)
        acc_train.append(accuracy)
        # compute test and validation error and accuracy
        error_valid.append(error_func(W, b, val_data[0], val_data[1], reg))
        error_test.append(error_func(W, b, test_data[0], test_data[1], reg))
        acc_valid.append(compute_accuracy(W, b, val_data[0], val_data[1]))
        acc_test.append(compute_accuracy(W, b, test_data[0], test_data[1]))
        # adjust the weight and biases
        old_w=np.concatenate((W.reshape(-1,1),np.array([b]).reshape(-1,1)))
        W = W - alpha * W_grad
        b = b - alpha * b_grad
        # stop the training loop if the error is lower than the error_tolerance
        #print("Epoch error ", epoch_error)
        if np.abs(np.linalg.norm(old_w)-np.linalg.norm(np.concatenate((W.reshape(-1,1),np.array([b]).reshape(-1,1))))) <= error_tol:
            break
    plot_trend([error_train, error_valid, error_test],
               data_title="Loss of Train, Validation and Test Data with alpha = 0.0001, lambda = 0",
               y_label="Loss")
    plot_trend([acc_train, acc_valid, acc_test], data_title="", y_label="Accuracy")
    return W, b

def MSE_normalEQ(W, b, x, y):
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[1])
    ones = np.ones((x.shape[0], 1))
    # construct an augmented data matrix that starts with a column of ones
    x_p = np.concatenate((ones, x), axis=1)
    W = (np.linalg.inv(x_p.T.dot(x_p)).dot(x_p.T)).dot(y)
    # extract the bias from the augmented weight vector
    b = W[0][0]
    # extract the desired weight vector from the augmented weight vector
    W = W[1:].squeeze()
    return W, b

def buildGraph(loss="MSE"):
    torch.manual_seed(421)
    error_train = []
    acc_train = []
    error_valid = []
    acc_valid = []
    error_test = []
    acc_test = []
    train_dataset = Dataset(torch.tensor(trainData).float(), torch.tensor(trainTarget).float())
    valid_dataset = Dataset(torch.tensor(validData).float(), torch.tensor(validTarget).float())
    test_dataset = Dataset(torch.tensor(testData).float(), torch.tensor(testTarget).float())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=500, shuffle=True)
    model = Prceptron(trainData.shape[1]*trainData.shape[1])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=0)
    loss_func = None
    if loss == "MSE":
        loss_func = torch.nn.MSELoss()
    elif loss == "CE":
        loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(0, 500):
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            feat, batch_label = batch
            feat = feat.type(torch.FloatTensor)
            result = model(feat)
            batch_label = batch_label.squeeze().float()
            loss = loss_func(input=result.squeeze(), target=batch_label.float())
            loss.backward()
            optimizer.step()
        error_train.append(compute_loss_with_tensor(model, train_dataset, loss_func))
        error_valid.append(compute_loss_with_tensor(model, valid_dataset, loss_func))
        error_test.append(compute_loss_with_tensor(model, test_dataset, loss_func))
        acc_train.append(evaluate(model, train_dataset))
        acc_valid.append(evaluate(model, valid_dataset))
        acc_test.append(evaluate(model, test_dataset))
    plot_trend([error_train, error_valid, error_test],
               data_title="pytorch_CE_lr==0.0001_weightDecay==0_Loss", y_label="Loss")
    plot_trend([acc_train, acc_valid, acc_test], data_title="pytorch_CE_lr==0.0001_weightDecay==0_Accuracy")
    weights = model.fc1.weight.data.detach().numpy()
    bias = model.fc1.bias.data.detach().numpy()
    return weights, bias

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
# print(trainData.shape) # (num of item, length, width)
def run_part_1():
    W = np.random.randint(1, size=(28 * 28,))
    W = np.random.random((28 * 28,))
    b = 0.5
    reg = 0
    epoch = 100
    lossType='CE'
    lossTypeDic={'MSE':gradMSE, 'CE':gradCE}
    errorTypeDic = {'MSE': MSE, 'CE': CrossEntropyLoss}
    loss_func=lossTypeDic[lossType]
    error_func=errorTypeDic[lossType]
    W_ana, b_ana = MSE_normalEQ(W, b, trainData, trainTarget)
    print(compute_accuracy(W_ana, b_ana, trainData, trainTarget))
    print(MSE(W_ana, b_ana, trainData, trainTarget, 0))
    W = np.random.randint(1, size=(28 * 28,))
    b = 0.5
    W, b = grad_descent(W, b, trainData, trainTarget, 0.005, epoch, reg, error_tol=0.0000001,
                        val_data=[validData, validTarget], test_data=[testData, testTarget],lossType=lossType)
    print(compute_accuracy(W, b, trainData, trainTarget))
    print(error_func(W, b, trainData, trainTarget, 0))
    lossType="MSE"
    loss_func=lossTypeDic[lossType]
    error_func=errorTypeDic[lossType]
    W = np.random.randint(1, size=(28 * 28,))
    b = 0.5
    W, b = grad_descent(W, b, trainData, trainTarget, 0.005, epoch, reg, error_tol=0.0000001,
                        val_data=[validData, validTarget], test_data=[testData, testTarget],lossType=lossType)
    print(compute_accuracy(W, b, trainData, trainTarget))
    print(error_func(W, b, trainData, trainTarget, 0))

# for reg in [0.001, 0.1, 0.5]:
#     W = np.random.randint(1, size=(28 * 28,))
#     b = 0.5
#     W, b = grad_descent(W, b, trainData, trainTarget, 0.0005, 5000, reg, error_tol=0.0000001,
#                         val_data=[validData, validTarget], test_data=[testData, testTarget])
#
#     print("the normal of the weight vector is" + str(np.linalg.norm(W)))

if __name__== "__main__":
    # run_part_1()
    print(buildGraph("CE"))
