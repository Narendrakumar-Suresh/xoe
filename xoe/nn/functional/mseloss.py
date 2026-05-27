from xoe.tensor import Tensor
from xoe.nn.module import Module


class MSELoss(Module):
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        squared_diff = diff * diff

        if self.reduction == "mean":
            return squared_diff.mean()
        elif self.reduction == "sum":
            return squared_diff.sum()

        return squared_diff
