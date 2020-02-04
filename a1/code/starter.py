import tensorflow as tf
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import time


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
    y_est = x.dot(w) + b * I
    #loss = np.average(y*np.log(1+np.exp(-y_est))+(1-y)*np.log(1+np.exp(y_est))) + reg / 2 * np.linalg.norm(w)
    loss = np.average(np.log(1+np.exp(-y*y_est)))
    pass
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
    w_grad = np.average(w_grad,0)
    b_grad = grad
    b_grad = np.average(b_grad,0)
    return b_grad,w_grad


def compute_accuracy(W, b, x, y):
    try:
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[1])
    except:
        print("nothing")
    output = x.dot(W) + b
    # print(output.tolist())
    # A[2]
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
        W = W - alpha * W_grad
        b = b - alpha * b_grad
        # stop the training loop if the error is lower than the error_tolerance
        #print("Epoch error ", epoch_error)
        if epoch_error <= error_tol:
            break
    plot_trend([error_train, error_valid, error_test],
               data_title="Loss of Train, Validation and Test Data with alpha = 0.0005, lambda = 0.5", y_label="Loss")
    plot_trend([acc_train, acc_valid, acc_test])
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
    # Initialize weight and bias tensors
    # tf.set_random_seed(421)
    # if loss == "MSE":
    #     b = 3
    # elif loss == "CE":
    #     a = 2
    pass


trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
# print(trainData.shape) # (num of item, length, width)
np.random.seed(20)
W = np.random.random((28*28, ))
b = 0
reg = 0.5
# W = np.random.randint(1, size=(28 * 28,))
# W = np.random.random((28 * 28,))
# b = 0.5
# reg = 0.1
lossType='CE'
lossTypeDic={'MSE':gradMSE, 'CE':gradCE}
loss_func=lossTypeDic[lossType]
W_ana, b_ana = MSE_normalEQ(W, b, trainData, trainTarget)
print(compute_accuracy(W_ana, b_ana, trainData, trainTarget))
print(loss_func(W_ana, b_ana, trainData, trainTarget, 0))
W = np.random.randint(1, size=(28 * 28,))
b = 0.5
# W, b = grad_descent(W, b, trainData, trainTarget, 0.00005, 100, reg, error_tol = 0.0000001, val_data = [validData, validTarget], test_data=[testData, testTarget])
# print(compute_accuracy(W, b, trainData, trainTarget))
# print(MSE(W, b, trainData, trainTarget, 0))
W, b = grad_descent(W, b, trainData, trainTarget, 0.0005, 5000, reg, error_tol=0.0000001,
                    val_data=[validData, validTarget], test_data=[testData, testTarget],lossType='CE')
print(compute_accuracy(W, b, trainData, trainTarget))
print(CrossEntropyLoss(W, b, trainData, trainTarget, 0))

# for reg in [0.001, 0.1, 0.5]:
#     W = np.random.randint(1, size=(28 * 28,))
#     b = 0.5
#     W, b = grad_descent(W, b, trainData, trainTarget, 0.0005, 5000, reg, error_tol=0.0000001,
#                         val_data=[validData, validTarget], test_data=[testData, testTarget])
#
#     print("the normal of the weight vector is" + str(np.linalg.norm(W)))
