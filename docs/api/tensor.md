# Tensor

The `Tensor` class wraps a JAX array and optionally tracks gradients for automatic differentiation.

## Constructor

```python
Tensor(data, requires_grad=False, dtype=None)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `data` | list, array, or Tensor | -- | Initial data |
| `requires_grad` | bool | `False` | Whether to track gradients |
| `dtype` | dtype | `None` (uses default) | Data type (float32, float16, bfloat16) |

## Properties

### `.dtype`

Returns the dtype of the underlying JAX array.

### `.shape`

Returns the shape tuple.

### `.T`

Returns a new `Tensor` with transposed data.

### `.grad`

The gradient `Tensor` computed by `backward()`. `None` until `backward()` is called.

## Methods

### `.reshape(*shape)`

```python
a.reshape(4, 4)
a.reshape(-1)
```

### `.squeeze(axis=None)`

Remove dimensions of size 1. If `axis` is given, only remove that dimension.

```python
a.squeeze()
a.squeeze(axis=0)
```

### `.unsqueeze(axis)`

Add a dimension at the given axis.

```python
a.unsqueeze(axis=0)
a.unsqueeze(axis=-1)
```

### `.sum(axis=None, keepdims=False)`

### `.mean(axis=None, keepdims=False)`

### `.max(axis=None, keepdims=False)`

### `.exp()`

Element-wise exponential.

### `.log()`

Element-wise natural logarithm.

### `.tanh()`

Element-wise hyperbolic tangent.

## Operators

| Operator | Method |
|---|---|
| `a + b` | `__add__` |
| `b + a` | `__radd__` |
| `a - b` | `__sub__` |
| `b - a` | `__rsub__` |
| `a * b` | `__mul__` |
| `b * a` | `__rmul__` |
| `a / b` | `__truediv__` |
| `b / a` | `__rtruediv__` |
| `a @ b` | `__matmul__` |
| `b @ a` | `__rmatmul__` |
| `a ** n` | `__pow__` |
| `-a` | `__neg__` |

## Comparisons

| Operator | Method |
|---|---|
| `a == b` | `__eq__` |
| `a != b` | `__ne__` |
| `a < b` | `__lt__` |
| `a <= b` | `__le__` |
| `a > b` | `__gt__` |
| `a >= b` | `__ge__` |

## Indexing

```python
a[0]
a[:, 1]
a[0:2]
```

## String Representation

```python
repr(a)  # Tensor([...], requires_grad=True)
```
