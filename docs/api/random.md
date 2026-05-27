# Random API

Utility functions for random number generation. xoe manages a **global JAX PRNG key** internally, so you don't have to thread keys through your code.

> **JAX PRNG primer**: Unlike NumPy/PyTorch, JAX uses **splittable PRNG keys** rather than a global mutable state. xoe wraps this so you can call `randn(shape)` without managing keys yourself.

---

## seed

```python
seed(n)
```

Initialize the global PRNG key with integer `n`. Call this **once at the start** of your program for reproducible results.

```python
from xoe.random import seed

seed(42)       # reproducible randomness
```

If you don't call `seed()`, the key starts with seed `0`.

---

## next_key

```python
next_key()
```

Split the global PRNG key and return a **new subkey**. This is used internally by layers like `Dropout` and `Linear` to get unique randomness each time they're called or initialized.

```python
from xoe.random import next_key

key = next_key()      # get a fresh subkey
```

You generally won't need to call this directly — it's used internally.

---

## randn

```python
randn(shape, key, dtype=None)
```

Returns a JAX array of random normal samples (mean 0, variance 1).

| Argument | Type | Default | Description |
|---|---|---|---|
| `shape` | `tuple` | required | Shape of the output array |
| `key` | JAX PRNG key | required | PRNG key (get one from `next_key()`) |
| `dtype` | dtype | `None` → `float32` | Data type of the output |

```python
from xoe.random import randn, next_key

a = randn((3, 4), next_key())        # 3x4 normal random array
b = randn((2, 2), next_key(), dtype=jnp.bfloat16)  # bfloat16
```

`randn` returns a raw `jax.Array`. Wrap it in `Tensor(...)` if you need a Tensor.

---

## zeros

```python
zeros(shape, dtype=None)
```

Returns a JAX array filled with zeros.

| Argument | Type | Default | Description |
|---|---|---|---|
| `shape` | `tuple` | required | Shape of the output array |
| `dtype` | dtype | `None` → `float32` | Data type |

```python
from xoe.random import zeros

b = zeros((2, 2))      # [[0., 0.], [0., 0.]]
```

---

## ones

```python
ones(shape, dtype=None)
```

Returns a JAX array filled with ones.

```python
from xoe.random import ones

c = ones((2, 2))       # [[1., 1.], [1., 1.]]
```

---

## Complete Example

```python
from xoe import Tensor
from xoe.random import seed, randn, zeros, ones, next_key

seed(0)                          # reproducible

key = next_key()
a = Tensor(randn((3, 4), key))   # 3x4 normal random Tensor

b = zeros((2, 2))                # raw JAX array of zeros
c = ones((2, 2))                 # raw JAX array of ones
```

---

## Why use `next_key()` with `randn()`?

JAX requires a fresh key for every random call. The pattern is:

```python
from xoe.random import randn, next_key

a = randn((3, 4), next_key())    # call next_key() to get a key
b = randn((3, 4), next_key())    # get another key for the next call
```

Internally, `next_key()` splits the global key deterministically, so `randn` results are reproducible when you set `seed()`.
