import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node

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


def zeros(shape, dtype=float32):
    return jnp.zeros(shape, dtype=dtype)


def ones(shape, dtype=float32):
    return jnp.ones(shape, dtype=dtype)


class Tensor:
    def __init__(self, data, dtype=None):
        if isinstance(data, Tensor):
            data = data._data
        if dtype is None:
            if isinstance(data, jnp.ndarray):
                self._data = data
            else:
                self._data = jnp.array(data, dtype=float32)
        else:
            self._data = jnp.array(data, dtype=dtype)
        self.grad = None

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def T(self):
        return Tensor._wrap(self._data.T)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @staticmethod
    def _wrap(data):
        t = object.__new__(Tensor)
        t._data = data
        return t

    def numpy(self):
        return np.asarray(self._data)

    def item(self):
        return self._data.item()

    def tolist(self):
        return np.asarray(self._data).tolist()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return Tensor._wrap(self._data.reshape(shape))

    def squeeze(self, axis=None):
        return Tensor._wrap(jnp.squeeze(self._data, axis=axis))

    def unsqueeze(self, axis):
        return Tensor._wrap(jnp.expand_dims(self._data, axis=axis))

    def detach(self):
        return Tensor._wrap(self._data)

    def sum(self, axis=None, keepdims=False):
        return Tensor._wrap(jnp.sum(self._data, axis=axis, keepdims=keepdims))

    def mean(self, axis=None, keepdims=False):
        return Tensor._wrap(jnp.mean(self._data, axis=axis, keepdims=keepdims))

    def max(self, axis=None, keepdims=False):
        return Tensor._wrap(jnp.max(self._data, axis=axis, keepdims=keepdims))

    def min(self, axis=None, keepdims=False):
        return Tensor._wrap(jnp.min(self._data, axis=axis, keepdims=keepdims))

    def __getitem__(self, idx):
        return Tensor._wrap(self._data[idx])

    def __len__(self):
        return len(self._data)

    def exp(self):
        return Tensor._wrap(jnp.exp(self._data))

    def log(self):
        return Tensor._wrap(jnp.log(self._data))

    def tanh(self):
        return Tensor._wrap(jnp.tanh(self._data))

    def __add__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor._wrap(self._data + other_data)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor._wrap(self._data - other_data)

    def __rsub__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor._wrap(other_data - self._data)

    def __mul__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor._wrap(self._data * other_data)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor._wrap(self._data / other_data)

    def __rtruediv__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor._wrap(other_data / self._data)

    def __matmul__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor._wrap(self._data @ other_data)

    def __rmatmul__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor._wrap(other_data @ self._data)

    def __pow__(self, power):
        return Tensor._wrap(self._data**power)

    def __neg__(self):
        return Tensor._wrap(-self._data)

    def __eq__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor._wrap(self._data == other_data)

    def __ne__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor._wrap(self._data != other_data)

    def __lt__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor._wrap(self._data < other_data)

    def __le__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor._wrap(self._data <= other_data)

    def __gt__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor._wrap(self._data > other_data)

    def __ge__(self, other):
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor._wrap(self._data >= other_data)

    def __bool__(self):
        if self._data.size != 1:
            raise ValueError(
                f"The truth value of a Tensor with {self._data.size} elements is ambiguous"
            )
        return bool(self._data)

    def __repr__(self):
        return f"Tensor({self._data.__repr__()})"

    def __hash__(self):
        if self._data.size == 1:
            return hash(self._data.item())
        return hash(self._data.tobytes())


def _tensor_flatten(t):
    return (t._data,), ()


def _tensor_unflatten(aux, children):
    return Tensor._wrap(children[0])


register_pytree_node(Tensor, _tensor_flatten, _tensor_unflatten)
