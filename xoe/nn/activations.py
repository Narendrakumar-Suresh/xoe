import jax.numpy as jnp
from xoe.tensor import Tensor

def relu(x:Tensor)->Tensor:
    return Tensor(jnp.maximum(
        0,x._data
    ))

def sigmoid(x:Tensor)->Tensor:
    return Tensor(
        1/(1+jnp.exp(-x._data))
    )

def softmax(x: Tensor, axis=-1) -> Tensor:
    return Tensor(jnp.exp(x._data) / jnp.sum(jnp.exp(x._data), axis=axis, keepdims=True))

def gelu(x: Tensor) -> Tensor:
    return Tensor(jnp.array(0.5) * x._data * (1 + jnp.tanh(jnp.sqrt(jnp.array(2 / jnp.pi)) * (x._data + 0.044715 * x._data**3))))