import jax
import jax.numpy as jnp
from tensor import Tensor

_key=jax.random.PRNGKey(0)

def next_key():
    global _key
    _key,subkey=jax.random.split(_key)
    return subkey

def seed(n):
    global _key
    _key = jax.random.PRNGKey(n)

def randn(shape):
    return Tensor(jax.random.normal(_key,shape))

def zeros(shape):
    return Tensor(jnp.zeros(shape))

def ones(shape):
    return Tensor(jnp.ones(shape))