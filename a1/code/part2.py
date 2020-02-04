import numpy as np

def CrossEntropyLoss(w, b, x, y, reg):
    N = x.shape[0]
    d = x.shape[1]
    I = np.ones(N)
    y_est = x.dot(w) + b*I
    loss = np.average(-y*np.log(y_est)-(I-y)*np.log(I-y_est))+reg/2*np.linalg.norm(w)
    return loss


def gradCE(w, b, x, y, reg):
    N = x.shape[0]
    d = x.shape[1]
    I = np.ones(N)
    y_est = x.dot(w) + b * I
    grad=-y/(1+np.exp(y_est*y))
    grad=np.array([grad]).T
    grad=-grad*x
    return grad

def grad_descent(W, b, x, y, alpha=0.005, epochs=5000, reg=0.1, error_tol = 0.0000001, val_data = [], test_data = [],lossType='MSE'):
    # initialize storage elements
    error_train = []
    acc_train = []
    error_valid = []
    acc_valid = []
    error_test = []
    acc_test = []
    if lossType=''
    for epoch in range(0, epochs):
        # compute gradient
        b_grad, W_grad = gradMSE(W, b, x, y, reg)
        # compute training error and accuracy
        accuracy = compute_accuracy(W, b, x, y)
        epoch_error = (MSE(W, b, x, y, reg))
        error_train.append(epoch_error)
        acc_train.append(accuracy)
        # compute test and validation error and accuracy
        error_valid.append(MSE(W, b, val_data[0], val_data[1], reg))
        error_test.append(MSE(W, b, test_data[0], test_data[1], reg))
        acc_valid.append(compute_accuracy(W, b, val_data[0], val_data[1]))
        acc_test.append(compute_accuracy(W, b, test_data[0], test_data[1]))
        # adjust the weight and biases
        W = W - alpha*W_grad
        b = b - alpha*b_grad
        # stop the training loop if the error is lower than the error_tolerance
        if epoch_error <=  error_tol:
            break
    plot_trend([error_train, error_valid, error_test], data_title="Loss of Train, Validation and Test Data with alpha = 0.0005, lambda = 0.5", y_label="Loss")
    plot_trend([acc_train, acc_valid, acc_test])
    return W,b