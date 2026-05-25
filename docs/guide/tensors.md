# Tensors

The `Tensor` class is the core data structure in xoe. It wraps a JAX array and optionally tracks gradients.

## Creation

```python
from xoe import Tensor

a = Tensor([1.0, 2.0, 3.0])
b = Tensor([[1.0, 2.0], [3.0, 4.0]])
c = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
```

## Operations

All standard arithmetic operators are supported:

```python
a + b   # __add__
a - b   # __sub__
a * b   # __mul__
a / b   # __truediv__
a @ b   # __matmul__
a ** 2  # __pow__
-a      # __neg__
```

Comparison operators return a `Tensor` of booleans:

```python
a == b
a != b
a <  b
a <= b
a >  b
a >= b
```

## Transpose

```python
a.T
```

## Shape Operations

```python
a.reshape(3, 1)
a.squeeze()          # remove all size-1 dims
a.squeeze(axis=0)    # remove specific size-1 dim
a.unsqueeze(axis=0)  # add a dimension
```

## Reductions

```python
a.sum()
a.sum(axis=0)
a.sum(axis=0, keepdims=True)

a.mean()
a.mean(axis=-1)

a.max()
a.max(axis=1)
```

## Element-wise Ops

```python
a.exp()
a.log()
a.tanh()
```

## Dtypes

xoe supports three floating-point dtypes:

```python
from xoe import float32, float16, bfloat16

Tensor([1.0], dtype=float32)
Tensor([1.0], dtype=float16)
Tensor([1.0], dtype=bfloat16)
```

### set_default_dtype

Set the default dtype for all new tensors:

```python
from xoe import set_default_dtype, bfloat16

set_default_dtype(bfloat16)
```

The default is `float32`.
