from xoe.nn.module import Module
from xoe.tensor import Tensor
from xoe import random


class Linear(Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, dtype=None
    ):
        self.W = Tensor(
            random.randn((in_features, out_features), dtype=dtype), requires_grad=True
        )
        self.b = (
            Tensor(random.zeros((out_features,), dtype=dtype), requires_grad=True)
            if bias
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out
