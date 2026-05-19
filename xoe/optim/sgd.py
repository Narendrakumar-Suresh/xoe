from xoe.nn.module import Module

class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p._data = p._data - self.lr * p.grad._data

    def zero_grad(self):
        for p in self.params:
            p.grad = None