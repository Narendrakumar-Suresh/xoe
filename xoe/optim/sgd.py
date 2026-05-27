import jax.numpy as jnp


class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocities = [None] * len(params)

    def step(self, params_data, grads):
        new_params = []
        for i, p in enumerate(params_data):
            g = grads[i]

            if self.weight_decay != 0:
                g = g + self.weight_decay * p

            if self.momentum != 0:
                if self.velocities[i] is None:
                    self.velocities[i] = jnp.zeros_like(p)
                self.velocities[i] = self.momentum * self.velocities[i] + g
                if self.nesterov:
                    g = g + self.momentum * self.velocities[i]
                else:
                    g = self.velocities[i]

            new_params.append(p - self.lr * g)
        return new_params
