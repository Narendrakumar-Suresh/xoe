import jax.numpy as jnp
from xoe.tensor import Tensor
from xoe.nn.module import Module


class CrossEntropyLoss(Module):
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        logits_max = jnp.max(logits.data, axis=-1, keepdims=True)
        shifted_logits = logits.data - logits_max

        log_softmax = shifted_logits - jnp.log(
            jnp.sum(jnp.exp(shifted_logits), axis=-1, keepdims=True)
        )

        if jnp.issubdtype(targets.data.dtype, jnp.integer):
            loss = -log_softmax[jnp.arange(logits.data.shape[0]), targets.data]
        elif targets.data.ndim < logits.data.ndim:
            loss = -log_softmax[
                jnp.arange(logits.data.shape[0]), targets.data.astype(jnp.int32)
            ]
        else:
            loss = -jnp.sum(targets.data * log_softmax, axis=-1)

        if self.reduction == "mean":
            return Tensor._wrap(jnp.mean(loss))
        elif self.reduction == "sum":
            return Tensor._wrap(jnp.sum(loss))
        return Tensor._wrap(loss)
