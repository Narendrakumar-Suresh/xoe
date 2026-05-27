import jax.numpy as jnp


class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocities = [None] * len(self.params)

    def step(self):
        for i, p in enumerate(self.params):
            g = p.grad.data
            data = p.data

            if self.weight_decay != 0:
                g = g + self.weight_decay * data

            if self.momentum != 0:
                if self.velocities[i] is None:
                    self.velocities[i] = jnp.zeros_like(data)
                self.velocities[i] = self.momentum * self.velocities[i] + g
                if self.nesterov:
                    g = g + self.momentum * self.velocities[i]
                else:
                    g = self.velocities[i]

            p.data = data - self.lr * g

    def zero_grad(self):
        for p in self.params:
            p.grad = None
