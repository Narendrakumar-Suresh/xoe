from __future__ import annotations

import jax.numpy as jnp

from xoe import random
from xoe.tensor import Tensor
from xoe.nn.module import Module


class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype=None,
        padding_idx: int | None = None,
        key=None,
    ):
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.padding_idx = padding_idx

        scale = 1.0 / (embedding_dim**0.5)
        k = key if key is not None else random.next_key()
        self.weight = Tensor(
            scale
            * random.randn(
                (self.num_embeddings, self.embedding_dim), key=k, dtype=dtype
            ),
        )

    def forward(self, idx: Tensor | jnp.ndarray) -> Tensor:
        if isinstance(idx, Tensor):
            idx = idx.data

        idx = jnp.asarray(idx, dtype=jnp.int32)

        if self.padding_idx is not None:
            mask = idx != self.padding_idx
            out = self.weight.data[idx]
            out = out * mask[..., None]
            return Tensor._wrap(out)

        return Tensor._wrap(self.weight.data[idx])
