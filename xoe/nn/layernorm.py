import jax.numpy as jnp
from xoe.tensor import Tensor
from xoe.nn.module import Module

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = Tensor(jnp.ones(normalized_shape), requires_grad=True)
        self.bias = Tensor(jnp.zeros(normalized_shape), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x._data.mean(axis=dims, keepdims=True)
        var = x._data.var(axis=dims, keepdims=True)
        x_hat = (x._data - mean) / jnp.sqrt(var + self.eps)
        out = x_hat * self.weight._data + self.bias._data
        return Tensor(out)
