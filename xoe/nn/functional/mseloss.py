from xoe.tensor import Tensor

class MSELoss:
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, pred: Tensor, target: Tensor):
        diff = pred - target
        squared_diff = diff * diff

        if self.reduction == "mean":
            return squared_diff.mean()
        elif self.reduction == "sum":
            return squared_diff.sum()

        return squared_diff
