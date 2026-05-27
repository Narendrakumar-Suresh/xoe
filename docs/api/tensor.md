# Tensor API

The `Tensor` class wraps a JAX array (`jax.Array`) and can optionally hold a gradient after `backward()` is called.

---

## Constructor

```python
Tensor(data, dtype=None)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `data` | list, `jnp.ndarray`, or `Tensor` | required | Initial data. Passing a `Tensor` copies its data. |
| `dtype` | dtype | `None` → `float32` | Data type. See [Dtypes](#dtypes) below. |

```python
from xoe import Tensor

Tensor([1.0, 2.0, 3.0])
Tensor([[1.0, 2.0], [3.0, 4.0]])
Tensor(jnp.array([1.0, 2.0]))       # from JAX array
Tensor(Tensor([1.0, 2.0]))           # from another Tensor
Tensor([1.0, 2.0], dtype=bfloat16)  # custom dtype
```

---

## Properties

| Property | Type | Description |
|---|---|---|
| `.data` | `jax.Array` | Get or set the underlying JAX array |
| `.grad` | `Tensor` or `None` | Gradient set by `backward()`. `None` before `backward()` is called. |
| `.shape` | `tuple` | Shape of the tensor (e.g. `(2, 3)`) |
| `.ndim` | `int` | Number of dimensions (e.g. `2`) |
| `.dtype` | `jnp.dtype` | Data type (e.g. `jnp.float32`) |
| `.T` | `Tensor` | Transposed tensor (new view, not a copy) |

```python
a = Tensor([[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]])

a.data    # jax.Array([[1., 2., 3.], [4., 5., 6.]])
a.grad    # None (until backward() is called)
a.shape   # (2, 3)
a.ndim    # 2
a.dtype   # jnp.float32
a.T       # Tensor with shape (3, 2)
```

---

## Methods

### `.item()`

Convert a 0-d or 1-element tensor to a Python scalar.

```python
Tensor([42.0]).item()   # 42.0
```

### `.tolist()`

Convert to a (possibly nested) Python list.

```python
Tensor([1.0, 2.0]).tolist()   # [1.0, 2.0]
```

### `.numpy()`

Convert to a NumPy array.

```python
a.numpy()   # np.ndarray
```

### `.detach()`

Return a new Tensor that shares the same underlying data (no gradient attached).

```python
a.detach()   # Tensor with same data, .grad = None
```

### `.reshape(*shape)`

Return a new tensor with the given shape. One dimension may be `-1` to infer.

```python
a.reshape(6)       # flatten to 1-D
a.reshape(3, 2)    # reshape
a.reshape(-1)      # flatten (inferred size)
```

### `.squeeze(axis=None)`

Remove dimensions of size 1. If `axis` is given, only remove that specific dimension.

```python
a = Tensor([[1.0, 2.0, 3.0]])  # shape (1, 3)
a.squeeze()                     # shape (3,)
a.squeeze(axis=0)               # shape (3,)
```

### `.unsqueeze(axis)`

Add a dimension of size 1 at the given position.

```python
a = Tensor([1.0, 2.0, 3.0])    # shape (3,)
a.unsqueeze(0)                  # shape (1, 3)
a.unsqueeze(-1)                 # shape (3, 1)
```

### `.sum(axis=None, keepdims=False)`

Sum of all elements (if `axis=None`) or along `axis`.

```python
a.sum()
a.sum(axis=0)
a.sum(axis=0, keepdims=True)
```

### `.mean(axis=None, keepdims=False)`

Mean of all elements or along `axis`.

```python
a.mean()
a.mean(axis=-1)
```

### `.max(axis=None, keepdims=False)`

Maximum of all elements or along `axis`.

```python
a.max()
a.max(axis=1)
```

### `.min(axis=None, keepdims=False)`

Minimum of all elements or along `axis`.

```python
a.min()
a.min(axis=0)
```

### `.exp()`

Element-wise exponential (`e^x`).

### `.log()`

Element-wise natural logarithm (`ln(x)`).

### `.tanh()`

Element-wise hyperbolic tangent.

---

## Arithmetic Operators

| Operator | Method |
|---|---|
| `a + b` | `__add__` |
| `b + a` (where `b` is not a Tensor) | `__radd__` |
| `a - b` | `__sub__` |
| `b - a` (reverse) | `__rsub__` |
| `a * b` | `__mul__` |
| `b * a` (reverse) | `__rmul__` |
| `a / b` | `__truediv__` |
| `b / a` (reverse) | `__rtruediv__` |
| `a @ b` | `__matmul__` |
| `b @ a` (reverse) | `__rmatmul__` |
| `a ** n` | `__pow__` |
| `-a` | `__neg__` |

---

## Comparison Operators

All comparisons return a Tensor of booleans.

| Operator | Method |
|---|---|
| `a == b` | `__eq__` |
| `a != b` | `__ne__` |
| `a < b` | `__lt__` |
| `a <= b` | `__le__` |
| `a > b` | `__gt__` |
| `a >= b` | `__ge__` |

---

## Indexing

Standard NumPy-style indexing works:

```python
a[0]                       # first element
a[-1]                      # last element
a[0:2]                     # slice
a[:, 1]                    # all rows, column 1
a[0, :]                    # row 0, all columns
a[0:2, 1:3]                # 2D slice
a[..., 0]                  # ellipsis indexing
```

---

## Dtypes

```python
from xoe import float32, float16, bfloat16, int32, int16

# Aliases
f32 = float32
f16 = float16
bf16 = bfloat16
i32 = int32
i16 = int16
```

---

## Utility Functions

```python
from xoe import zeros, ones

zeros((3, 4))       # JAX array of zeros
ones((2, 2))        # JAX array of ones
```

Note: `zeros` and `ones` return raw JAX arrays, not Tensors. Use `Tensor(zeros(...))` if you need a Tensor.

---

## String Representation

```python
repr(a)   # "Tensor([1. 2. 3.])"
```
