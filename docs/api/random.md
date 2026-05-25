# Random

Utility functions for random number generation. xoe manages a global JAX PRNG key internally.

## seed

```python
seed(n)
```

Initialize the global PRNG key with integer `n`. Call this once at the start of your program.

```python
from xoe.random import seed

seed(42)
```

## next_key

```python
next_key()
```

Split the global PRNG key and return a subkey. Used internally by layers like `Dropout`.

## randn

```python
randn(shape, dtype=None)
```

Returns a JAX array of random normal samples. `dtype` defaults to the current default dtype.

## zeros

```python
zeros(shape, dtype=None)
```

Returns a JAX array filled with zeros.

## ones

```python
ones(shape, dtype=None)
```

Returns a JAX array filled with ones.

## Example

```python
from xoe.random import seed, randn, zeros, ones

seed(0)

a = randn((3, 4))     # normal random
b = zeros((2, 2))     # all zeros
c = ones((2, 2))      # all ones
```
