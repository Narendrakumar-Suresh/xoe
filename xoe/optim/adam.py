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
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.amsgrad = amsgrad
        self.t = 0
        self.m = [jnp.zeros_like(p.data) for p in params]
        self.v = [jnp.zeros_like(p.data) for p in params]
        self.v_max = [jnp.zeros_like(p.data) for p in params] if amsgrad else None

    def step(self, params_data, grads):
        self.t += 1
        b1, b2 = self.betas
        new_params = []
        for i, p in enumerate(params_data):
            g = grads[i]
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * (g**2)

            if self.amsgrad:
                self.v_max[i] = jnp.maximum(self.v_max[i], self.v[i])
                denom = jnp.sqrt(self.v_max[i]) + self.eps
            else:
                denom = jnp.sqrt(self.v[i]) + self.eps

            m_hat = self.m[i] / (1 - b1**self.t)
            new_params.append(p - self.lr * m_hat / denom)
        return new_params


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
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.t = 0
        self.m = [jnp.zeros_like(p.data) for p in params]
        self.v = [jnp.zeros_like(p.data) for p in params]
        self.v_max = [jnp.zeros_like(p.data) for p in params] if amsgrad else None

    def step(self, params_data, grads):
        self.t += 1
        b1, b2 = self.betas
        new_params = []
        for i, p in enumerate(params_data):
            g = grads[i]
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * (g**2)

            if self.amsgrad:
                self.v_max[i] = jnp.maximum(self.v_max[i], self.v[i])
                denom = jnp.sqrt(self.v_max[i]) + self.eps
            else:
                denom = jnp.sqrt(self.v[i]) + self.eps

            m_hat = self.m[i] / (1 - b1**self.t)
            p_new = p - self.lr * m_hat / denom
            p_new = p_new - self.lr * self.weight_decay * p_new
            new_params.append(p_new)
        return new_params
