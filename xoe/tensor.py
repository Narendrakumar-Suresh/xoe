import jax.numpy as jnp

# Global default dtype management
_default_dtype = jnp.float32

# Dtypes
float32 = jnp.float32
float16 = jnp.float16
bfloat16 = jnp.bfloat16
int32 = jnp.int32
int16 = jnp.int16

f32 = float32
f16 = float16
bf16 = bfloat16
i32 = int32
i16 = int16


def set_default_dtype(dtype):
    global _default_dtype
    _default_dtype = dtype


def get_default_dtype():
    return _default_dtype


class Tensor:
    def __init__(self, data, requires_grad=False, dtype=None):
        if dtype is None:
            dtype = _default_dtype
        if isinstance(data, Tensor):
            self._data = jnp.array(data._data, dtype=dtype)
        else:
            self._data = jnp.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad = None

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._data.shape

    @property
    def T(self):
        return Tensor(self._data.T, requires_grad=self.requires_grad)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return Tensor(self._data.reshape(shape), requires_grad=self.requires_grad)

    def squeeze(self, axis=None):
        return Tensor(
            jnp.squeeze(self._data, axis=axis), requires_grad=self.requires_grad
        )

    def unsqueeze(self, axis):
        return Tensor(
            jnp.expand_dims(self._data, axis=axis), requires_grad=self.requires_grad
        )

    # Reductions
    def sum(self, axis=None, keepdims=False):
        return Tensor(
            jnp.sum(self._data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
        )

    def mean(self, axis=None, keepdims=False):
        return Tensor(
            jnp.mean(self._data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
        )

    def max(self, axis=None, keepdims=False):
        return Tensor(
            jnp.max(self._data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
        )

    # Indexing
    def __getitem__(self, idx):
        return Tensor(self._data[idx], requires_grad=self.requires_grad)

    # Math ops
    def exp(self):
        return Tensor(jnp.exp(self._data), requires_grad=self.requires_grad)

    def log(self):
        return Tensor(jnp.log(self._data), requires_grad=self.requires_grad)

    def tanh(self):
        return Tensor(jnp.tanh(self._data), requires_grad=self.requires_grad)

    # Dunder operators
    def __add__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data + other_data, requires_grad=self.requires_grad)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data - other_data, requires_grad=self.requires_grad)

    def __rsub__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(other_data - self._data, requires_grad=self.requires_grad)

    def __mul__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data * other_data, requires_grad=self.requires_grad)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data / other_data, requires_grad=self.requires_grad)

    def __rtruediv__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(other_data / self._data, requires_grad=self.requires_grad)

    def __matmul__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data @ other_data, requires_grad=self.requires_grad)

    def __rmatmul__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(other_data @ self._data, requires_grad=self.requires_grad)

    def __pow__(self, power):
        return Tensor(self._data**power, requires_grad=self.requires_grad)

    def __neg__(self):
        return Tensor(-self._data, requires_grad=self.requires_grad)

    # Comparison ops (mostly element-wise returning boolean JAX arrays/tensors)
    def __eq__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data == other_data)

    def __ne__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data != other_data)

    def __lt__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data < other_data)

    def __le__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data <= other_data)

    def __gt__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data > other_data)

    def __ge__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data >= other_data)

    def __repr__(self):
        return f"Tensor({self._data.__repr__()}, requires_grad={self.requires_grad})"
