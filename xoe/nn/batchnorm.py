import jax.numpy as jnp
from xoe.tensor import Tensor
from xoe.nn.module import Module


class BatchNorm1d(Module):
    def __init__(self, features, eps=1e-5, momentum=0.1, affine=True):
        self.features = features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.training = True
        if affine:
            self.weight = Tensor(jnp.ones(features))
            self.bias = Tensor(jnp.zeros(features))
        self._running_mean = Tensor(jnp.zeros(features))
        self._running_var = Tensor(jnp.ones(features))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.data.mean(axis=0)
            var = x.data.var(axis=0)
            self._running_mean.data = (
                1 - self.momentum
            ) * self._running_mean.data + self.momentum * mean
            self._running_var.data = (
                1 - self.momentum
            ) * self._running_var.data + self.momentum * var
        else:
            mean = self._running_mean.data
            var = self._running_var.data

        x_hat = (x.data - mean) / jnp.sqrt(var + self.eps)
        if self.affine:
            out = x_hat * self.weight.data + self.bias.data
        else:
            out = x_hat
        return Tensor._wrap(out)
