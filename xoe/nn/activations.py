import jax.numpy as jnp
from xoe.tensor import Tensor
from xoe.nn.module import Module


def relu(x: Tensor) -> Tensor:
    return Tensor._wrap(jnp.maximum(0, x.data))


def sigmoid(x: Tensor) -> Tensor:
    return Tensor._wrap(1 / (1 + jnp.exp(-x.data)))


def softmax(x: Tensor, axis=-1) -> Tensor:
    x_max = jnp.max(x.data, axis=axis, keepdims=True)
    shifted = x.data - x_max
    exp_shifted = jnp.exp(shifted)
    return Tensor._wrap(exp_shifted / jnp.sum(exp_shifted, axis=axis, keepdims=True))


def leaky_relu(x: Tensor, negative_slope=0.01) -> Tensor:
    return Tensor._wrap(jnp.where(x.data > 0, x.data, negative_slope * x.data))


def gelu(x: Tensor) -> Tensor:
    return Tensor._wrap(
        0.5
        * x.data
        * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x.data + 0.044715 * x.data**3))),
    )


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)


class Softmax(Module):
    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return softmax(x, axis=self.axis)


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return leaky_relu(x, self.negative_slope)


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return gelu(x)
