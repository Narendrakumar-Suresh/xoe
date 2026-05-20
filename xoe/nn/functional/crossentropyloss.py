import jax.numpy as jnp
from xoe.tensor import Tensor
from xoe.nn.module import Module


class CrossEntropyLoss(Module):
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        logits_max = jnp.max(logits._data, axis=-1, keepdims=True)
        shifted_logits = logits._data - logits_max

        log_softmax = shifted_logits - jnp.log(
            jnp.sum(jnp.exp(shifted_logits), axis=-1, keepdims=True)
        )

        loss = -jnp.sum(targets._data * log_softmax, axis=-1)

        if self.reduction == "mean":
            return Tensor(jnp.mean(loss))
        elif self.reduction == "sum":
            return Tensor(jnp.sum(loss))
        return Tensor(loss)
