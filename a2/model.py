import numpy as np


class LinearModel:
    def __init__(self, d1, d2, K, act_func, loss_func, grad_lossfunc):
        self.in_size = d1
        self.hidden_size = d2
        self.K = K
        self.w_h = np.random.randn(d1, d2) * 2 / (d1 + d2)
        self.v_wh = np.ones((d1, d2)) * 0.00001
        self.w_o = np.random.randn(d2, K) * 2 / (d2 + K)
        self.v_wo = np.ones((d2, K)) * 0.00001
        self.b_h = np.random.randn(d2, 1) * 2 / (d2 + 1)
        self.v_bh = np.ones((d2, 1)) * 0.00001
        self.b_o = np.random.randn(K, 1) * 2 / (K + 1)
        self.v_bo = 0.00001
        self.act_func = act_func
        self.loss_func = loss_func
        self.grad_lossfunc = grad_lossfunc
        self.loss = 0
        self.x = None
        self.x_h = None
        self.x_h_a = None
        self.x_o = None
        self.x_o_a = None


    def grad_relu(self, x):
        return (x > 0).astype(int)

    def forward(self, x):
        x = x.reshape(-1, 1)
        self.x=x
        self.x_h = np.matmul(self.w_h.T, x) + self.b_h  # shape d2 * 1
        self.x_h_a = self.act_func[0](self.x_h)  # shape d2 * 1
        self.x_o = np.matmul(self.w_o.T, self.x_h_a) + self.b_o  # shape K * 1
        self.x_o_a = self.act_func[1](self.x_o)  # shape K * 1
        self.x_o_a.reshape(-1)  # shape K,
        return (self.x_o_a+0.00000000001)

    def compute_loss(self, target, x):
        self.loss = self.loss_func(target, x)
        return self.loss

    def loss_backward(self, target, gamma=0.99, alpha=0.00001):
        grad_loss = np.array(self.grad_lossfunc(target, self.x_o_a.reshape(-1))).reshape(-1, 1)  # shape K * 1
        grad_wo = np.matmul(grad_loss, self.x_h.T).T  # shape d2 * K
        grad_bo = grad_loss
        grad_wh = np.matmul(self.x,self.grad_relu(self.x_h).T*np.matmul(self.w_o,grad_loss).T)
        grad_bh = self.grad_relu(self.x_h)*np.matmul(self.w_o,grad_loss)
        self.v_wo = gamma * self.v_wo + alpha * grad_wo
        self.w_o -= self.v_wo
        self.v_bo = gamma * self.v_bo + alpha * grad_bo
        self.b_o -= self.v_bo
        self.v_wh = gamma * self.v_wh + alpha * grad_wh
        self.w_h -= self.v_wh
        self.v_bh = gamma * self.v_bh + alpha * grad_bh
        self.b_h -= self.v_bh
