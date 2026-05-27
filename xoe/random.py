import jax
from xoe.tensor import float32

_key = jax.random.PRNGKey(0)


def next_key(key=None):
    global _key
    if key is None:
        _key, subkey = jax.random.split(_key)
        return subkey
    k1, k2 = jax.random.split(key)
    return k2


def split_key(key=None, num=2):
    global _key
    if key is None:
        new_keys = jax.random.split(_key, num)
        _key = new_keys[0]
        return new_keys[1:] if num > 2 else new_keys[1]
    return jax.random.split(key, num)[1:] if num > 2 else jax.random.split(key, num)[1]


def seed(n):
    global _key
    _key = jax.random.PRNGKey(n)


def randn(shape, key, dtype=None):
    if dtype is None:
        dtype = float32
    return jax.random.normal(key, shape, dtype=dtype)
