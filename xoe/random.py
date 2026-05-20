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


from xoe.tensor import get_default_dtype


def randn(shape, dtype=None):
    if dtype is None:
        dtype = get_default_dtype()
    return jax.random.normal(next_key(), shape, dtype=dtype)  # raw array


def zeros(shape, dtype=None):
    if dtype is None:
        dtype = get_default_dtype()
    return jnp.zeros(shape, dtype=dtype)  # raw array


def ones(shape, dtype=None):
    if dtype is None:
        dtype = get_default_dtype()
    return jnp.ones(shape, dtype=dtype)  # raw array
