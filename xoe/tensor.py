import jax.numpy as jnp


class Tensor:
    def __init__(self, data, requires_grad=False):
        self._data = jnp.array(data)
        self.requires_grad = requires_grad
        self.grad = None

    @property
    def T(self):
        return Tensor(self._data.T)

    @property
    def shape(self):
        return self._data.shape

    def __matmul__(self, other):
        return Tensor(self._data @ other._data)

    def __add__(self, other):
        return Tensor(self._data + other._data)

    def __mul__(self, other):
        return Tensor(self._data * other._data)

    def __pow__(self, exp):
        return Tensor(self._data**exp)

    def __truediv__(self, other):
        return Tensor(self._data / other._data)

    def __sub__(self, other):
        return Tensor(self._data - other._data)

    def __repr__(self):
        return f"Tensor({self._data})"

    def squeeze(self, axis=None):
        return Tensor(jnp.squeeze(self._data, axis=axis))

    def unsqueeze(self, axis):
        return Tensor(jnp.expand_dims(self._data, axis=axis))

    def sum(self, axis=None):
        return Tensor(jnp.sum(self._data, axis=axis))

    def mean(self, axis=None):
        return Tensor(jnp.mean(self._data, axis=axis))

    def max(self, axis=None):
        return Tensor(jnp.max(self._data, axis=axis))

    def __eq__(self, other):
        return self._data == other._data

    def __ne__(self, other):
        return self._data != other._data

    def __lt__(self, other):
        return self._data < other._data

    def __gt__(self, other):
        return self._data > other._data

    def __ge__(self, other):
        return self._data >= other._data

    def __le__(self, other):
        return self._data <= other._data

    def reshape(self, *shape):
        return Tensor(self._data.reshape(shape))
