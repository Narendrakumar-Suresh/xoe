import jax.numpy as jnp


class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = [jnp.zeros_like(p._data) for p in params]
        self.v = [jnp.zeros_like(p._data) for p in params]

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is not None:
                g = p.grad._data
                self.m[i] = b1 * self.m[i] + (1 - b1) * g
                self.v[i] = b2 * self.v[i] + (1 - b2) * (g**2)

                m_hat = self.m[i] / (1 - b1**self.t)
                v_hat = self.v[i] / (1 - b2**self.t)

                p._data = p._data - self.lr * m_hat / (jnp.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class AdamW:
    def __init__(
        self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [jnp.zeros_like(p._data) for p in params]
        self.v = [jnp.zeros_like(p._data) for p in params]

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is not None:
                p._data = p._data - self.lr * self.weight_decay * p._data

                g = p.grad._data
                self.m[i] = b1 * self.m[i] + (1 - b1) * g
                self.v[i] = b2 * self.v[i] + (1 - b2) * (g**2)

                m_hat = self.m[i] / (1 - b1**self.t)
                v_hat = self.v[i] / (1 - b2**self.t)

                p._data = p._data - self.lr * m_hat / (jnp.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None
