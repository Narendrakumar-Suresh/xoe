import jax
import jax.numpy as jnp
from xoe.tensor import Tensor

_key = jax.random.PRNGKey(0)


def next_key():
    global _key
    _key, subkey = jax.random.split(_key)
    return subkey


def seed(n):
    global _key
    _key = jax.random.PRNGKey(n)


def randn(shape):
    return jax.random.normal(next_key(), shape)  # raw array

def zeros(shape):
    return jnp.zeros(shape)  # raw array

def ones(shape):
    return jnp.ones(shape)  # raw array