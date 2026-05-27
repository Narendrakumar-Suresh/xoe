import jax
from xoe.tensor import Tensor
from xoe.nn.module import Module
from xoe.random import next_key


class Dropout(Module):
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, x: Tensor, key=None) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        k = key if key is not None else next_key()
        mask = jax.random.bernoulli(k, p=1.0 - self.p, shape=x.data.shape)
        out = (x.data * mask) / (1.0 - self.p)
        return Tensor._wrap(out)
