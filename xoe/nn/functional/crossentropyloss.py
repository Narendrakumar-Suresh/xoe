import jax.numpy as jnp
from xoe.tensor import Tensor


class CrossEntropyLoss:
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        logits_max = jnp.max(logits._data, axis=-1, keepdims=True)
        shifted_logits = logits._data - logits_max

        log_softmax = shifted_logits - jnp.log(
            jnp.sum(jnp.exp(shifted_logits), axis=-1, keepdims=True)
        )

        if (
            jnp.issubdtype(targets._data.dtype, jnp.integer)
            or targets._data.ndim < logits._data.ndim
        ):
            if logits._data.ndim > 1:
                batch_size = logits._data.shape[0]
                loss = -log_softmax[
                    jnp.arange(batch_size), targets._data.astype(jnp.int32)
                ]
            else:
                loss = -log_softmax[targets._data.astype(jnp.int32)]
        else:
            loss = -jnp.sum(targets._data * log_softmax, axis=-1)

        if self.reduction == "mean":
            return Tensor(jnp.mean(loss))
        elif self.reduction == "sum":
            return Tensor(jnp.sum(loss))
        return Tensor(loss)
