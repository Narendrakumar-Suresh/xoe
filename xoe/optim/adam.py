import jax.numpy as jnp


class Adam:
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
    ):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.amsgrad = amsgrad
        self.t = 0
        self.m = [jnp.zeros_like(p.data) for p in self.params]
        self.v = [jnp.zeros_like(p.data) for p in self.params]
        self.v_max = [jnp.zeros_like(p.data) for p in self.params] if amsgrad else None

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            g = p.grad.data
            data = p.data
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * (g**2)

            if self.amsgrad:
                self.v_max[i] = jnp.maximum(self.v_max[i], self.v[i])
                denom = jnp.sqrt(self.v_max[i]) + self.eps
            else:
                denom = jnp.sqrt(self.v[i]) + self.eps

            m_hat = self.m[i] / (1 - b1**self.t)
            p.data = data - self.lr * m_hat / denom

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class AdamW:
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
    ):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.t = 0
        self.m = [jnp.zeros_like(p.data) for p in self.params]
        self.v = [jnp.zeros_like(p.data) for p in self.params]
        self.v_max = [jnp.zeros_like(p.data) for p in self.params] if amsgrad else None

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            g = p.grad.data
            data = p.data
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * (g**2)

            if self.amsgrad:
                self.v_max[i] = jnp.maximum(self.v_max[i], self.v[i])
                denom = jnp.sqrt(self.v_max[i]) + self.eps
            else:
                denom = jnp.sqrt(self.v[i]) + self.eps

            m_hat = self.m[i] / (1 - b1**self.t)
            p_new = data - self.lr * m_hat / denom
            p_new = p_new - self.lr * self.weight_decay * p_new
            p.data = p_new

    def zero_grad(self):
        for p in self.params:
            p.grad = None
