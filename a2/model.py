import numpy as np


class LinearModel:
    def __init__(self, d1, d2, K, act_func, loss_func, grad_lossfunc):
        self.w_h = np.random.randn(d1, d2) * 2 / (d1 + d2)
        self.v_wh = np.ones(d1, d2) * 0.00001
        self.w_o = np.random.randn(d2, K) * 2 / (d2 + K)
        self.v_wo = np.ones(d2, K) * 0.00001
        self.b_h = np.random.randn(d2, 1) * 2 / (d2 + 1)
        self.v_bh = np.ones(d2, 1) * 0.00001
        self.b_o = np.random.randn()
        self.v_bo = 0.00001
        self.act_func = act_func
        self.loss_func = loss_func
        self.grad_lossfunc = grad_lossfunc
        self.loss = 0
        self.x_h = None
        self.x_h_a = None
        self.x_o = None
        self.x_o_a = None

    def grad_linear(self, x):
        return x

    def grad_relu(x):
        return (x > 0).astype(int)

    def forward(self, x):
        self.x_h += np.matmul(self.w_h.T, x) + self.b_h
        self.x_h_a = self.act_func[0](self.x_h)
        self.x_o += np.matmul(self.w_o.T, self.outputs[-1]) + self.b_o
        self.x_o_a = self.act_func[1](self.x_o)
        return self.x_o_a

    def compute_loss(self, target, x):
        self.loss = self.loss_func(target, x)
        return self.loss

    def loss_backward(self, gamma=0.99, alpha=0.00001):
        grad_loss = self.grad_lossfunc(self.loss, self.x_o_a)
        grad_wo = grad_loss * self.x_o
        grad_bo = grad_loss
        grad_wh = grad_loss * self.grad_linear(self.x_o_a) * self.grad_relu(self.x_h) * self.x_h
        grad_bo = grad_loss * self.grad_linear(self.x_o_a) * self.grad_relu(self.x_h)
        self.v_wo = gamma * self.v_wo + alpha * grad_wo
        self.w_o -= self.v_wo
        self.v_bo = gamma * self.v_bo + alpha * grad_bo
        self.b_o -= self.v_bo
        self.v_wh = gamma * self.v_wh + alpha * grad_wh
        self.w_h -= self.v_wh
        self.v_wo = gamma * self.v_wo + alpha * grad_wo
        self.w_o -= self.v_wo
