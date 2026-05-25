import jax
from xoe.tensor import Tensor
from xoe.nn.module import Module
from xoe.random import next_key

class Dropout(Module):
    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        key = next_key()
        mask = jax.random.bernoulli(key, p=1.0 - self.p, shape=x._data.shape)
        out = (x._data * mask) / (1.0 - self.p)
        return Tensor(out)
