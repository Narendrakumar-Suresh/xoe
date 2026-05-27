import math
from xoe.tensor import Tensor, zeros
from xoe.nn.module import Module
from xoe import random


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype=None,
        key=None,
    ):
        bound = 1.0 / math.sqrt(in_features)
        k = key if key is not None else random.next_key()
        self.W = Tensor(
            random.randn((in_features, out_features), key=k, dtype=dtype) * bound,
        )
        self.b = Tensor(zeros((out_features,), dtype=dtype)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out
