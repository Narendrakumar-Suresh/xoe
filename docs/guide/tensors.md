# Tensors

The `Tensor` class is the core data structure in xoe. It wraps a JAX array (a `jax.Array`) and can optionally hold a gradient after `backward()` is called.

Think of a Tensor as a **smart array** — it behaves like a regular numeric array but also tracks the gradient needed for optimization.

---

## Creating Tensors

```python
from xoe import Tensor

# From a Python list
a = Tensor([1.0, 2.0, 3.0])

# From a nested list (2D tensor)
b = Tensor([[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]])

# From an existing JAX array
import jax.numpy as jnp
arr = jnp.array([1.0, 2.0, 3.0])
c = Tensor(arr)

# From another Tensor (copies the data)
d = Tensor(a)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `data` | list, `jnp.ndarray`, or `Tensor` | required | Initial data |
| `dtype` | dtype | `None` → `float32` | Data type (see Dtypes below) |

### Common shortcuts

```python
from xoe import zeros, ones

z = zeros((3, 4))    # 3x4 tensor of zeros
o = ones((2, 2))     # 2x2 tensor of ones
```

---

## Properties

### `.data` — the raw JAX array

Every Tensor wraps a JAX array. Access or replace it via `.data`:

```python
a = Tensor([1.0, 2.0, 3.0])
print(a.data)          # [1. 2. 3.]
print(type(a.data))    # <class 'jax.Array'>
a.data = jnp.array([4.0, 5.0, 6.0])  # replace the underlying data
```

### `.grad` — the gradient (set by `backward()`)

After you call `backward()`, each parameter's `.grad` is populated with a Tensor containing the gradient. Before that, it's `None`.

```python
loss, grads = backward(...)   # this is OLD style — xoe doesn't do this
# New style:
backward(loss_fn, params, x)  # populates p.grad on each param
print(params[0].grad)          # Tensor with gradient values
```

### `.shape`, `.ndim`, `.dtype`

```python
a = Tensor([[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]])

a.shape       # (2, 3)
a.ndim        # 2
a.dtype       # jnp.float32
```

### `.T` — transpose

```python
a.T            # shape (3, 2)
```

---

## Arithmetic Operations

All standard math operators work element-wise, just like in NumPy or PyTorch:

```python
a + b   # addition
a - b   # subtraction
a * b   # multiplication
a / b   # division
a @ b   # matrix multiplication (matmul)
a ** 2  # power
-a      # negation
```

xoe also handles **broadcasting** automatically, matching NumPy/JAX rules:

```python
a = Tensor([[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]])    # shape (2, 3)
b = Tensor([10.0, 20.0, 30.0])    # shape (3,) — broadcasts!

a + b  # shape (2, 3) — b is broadcast across rows
```

### Reverse (reflected) operators

These let you use a Tensor on the right side of an operation with a non-Tensor on the left:

```python
5 + a          # __radd__  → a + 5
5 - a          # __rsub__  → -(a - 5)
5 * a          # __rmul__  → a * 5
5 / a          # __rtruediv__
jnp.eye(3) @ a # __rmatmul__
```

---

## Comparison Operators

Comparisons return a Tensor of booleans:

```python
a == b
a != b
a <  b
a <= b
a >  b
a >= b
```

---

## Shape Operations

```python
a = Tensor([[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]])

a.reshape(3, 2)          # (2,3) → (3,2)
a.reshape(-1)            # (2,3) → (6,)  (-1 infers the dimension)
a.squeeze()              # remove all size-1 dimensions
a.squeeze(axis=0)        # remove a specific size-1 dimension
a.unsqueeze(axis=0)      # add a dimension: (2,3) → (1,2,3)
a.unsqueeze(axis=-1)     # add a dimension: (2,3) → (2,3,1)
```

---

## Reductions

```python
a.sum()                  # sum of all elements (scalar)
a.sum(axis=0)            # sum along axis 0
a.sum(axis=0, keepdims=True)  # keep the reduced dimension

a.mean()                 # mean of all elements
a.mean(axis=-1)          # mean along the last axis

a.max()                  # maximum of all elements
a.max(axis=1)            # max along axis 1

a.min()                  # minimum of all elements
a.min(axis=0)            # min along axis 0
```

---

## Element-wise Math

```python
a.exp()       # e^x
a.log()       # natural logarithm
a.tanh()      # hyperbolic tangent
```

---

## Utility Methods

```python
a = Tensor([1.0, 2.0, 3.0])

a.item()           # return as Python scalar (only for 0-d or 1-element tensors)
a.tolist()         # convert to Python list
a.numpy()          # convert to numpy array (np.asarray)
a.detach()         # return a new Tensor sharing the same data (no grad tracking)

# Indexing (same as NumPy/JAX)
a[0]               # first element
a[-1]              # last element
a[0:2]             # slice
b = Tensor([[1,2,3],[4,5,6]])
b[:, 1]            # column
b[0, :]            # row
```

---

## Dtypes

xoe supports three floating-point dtypes:

```python
from xoe import float32, float16, bfloat16

Tensor([1.0], dtype=float32)    # 32-bit float (default)
Tensor([1.0], dtype=float16)    # 16-bit float
Tensor([1.0], dtype=bfloat16)   # brain floating-point (good for TPUs)
```

Integer types are also available:

```python
from xoe import int32, int16

Tensor([1, 2, 3], dtype=int32)
```

The default dtype is `float32`.

---

## String Representation

```python
a = Tensor([1.0, 2.0, 3.0])
repr(a)   # "Tensor([1. 2. 3.])"
```

---

## Boolean Context

A Tensor with exactly one element can be used in a boolean context:

```python
if (a > 0).sum() > Tensor([0]):   # works — scalar comparison
    print("positive values exist")
```

A Tensor with more than one element raises an error if used in a boolean context (same as PyTorch).
