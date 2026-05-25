import jax.numpy as jnp
import jax
import math
from xoe.tensor import Tensor
from xoe.random import next_key


def zeros_(tensor: Tensor):
    tensor._data = jnp.zeros_like(tensor._data)
    return tensor


def ones_(tensor: Tensor):
    tensor._data = jnp.ones_like(tensor._data)
    return tensor


def constant_(tensor: Tensor, val):
    tensor._data = jnp.full_like(tensor._data, val)
    return tensor


def kaiming_uniform_(tensor: Tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    # Determine fan
    shape = tensor._data.shape
    if len(shape) >= 2:
        fan_in = shape[0]
    else:
        fan_in = shape[0] if len(shape) == 1 else 1

    if nonlinearity == "relu":
        gain = math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        gain = math.sqrt(2.0 / (1.0 + a**2))
    elif nonlinearity == "tanh":
        gain = 5.0 / 3.0
    else:
        gain = 1.0

    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    key = next_key()
    tensor._data = jax.random.uniform(
        key, shape=shape, minval=-bound, maxval=bound, dtype=tensor.dtype
    )
    return tensor


def xavier_uniform_(tensor: Tensor, gain=1.0):
    shape = tensor._data.shape
    if len(shape) >= 2:
        fan_in, fan_out = shape[0], shape[1]
    else:
        fan_in = shape[0] if len(shape) == 1 else 1
        fan_out = 1

    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    bound = math.sqrt(3.0) * std
    key = next_key()
    tensor._data = jax.random.uniform(
        key, shape=shape, minval=-bound, maxval=bound, dtype=tensor.dtype
    )
    return tensor
