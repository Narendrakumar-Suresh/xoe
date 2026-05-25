<p align="center">
  <img src="logo.png" alt="xoe" width="200">
</p>

**Write PyTorch. Run JAX.**

A deep learning framework with PyTorch-style imperative API backed by JAX and XLA. Stateful, fast, and built for TPUs.

```python
import xoe
from xoe import Tensor, backward
from xoe.nn import Linear, Sequential, LayerNorm
from xoe.nn.activations import GELU
from xoe.nn.functional import MSELoss
from xoe.optim import AdamW

model = Sequential(
    Linear(4, 16),
    GELU(),
    LayerNorm(16),
    Linear(16, 1),
)

x = Tensor([[1.0, 2.0, 3.0, 4.0]] * 8)
y = Tensor([[1.0]] * 8)

criterion = MSELoss()
optimizer = AdamW(model.parameters(), lr=1e-3)

for step in range(20):
    optimizer.zero_grad()
    def loss_fn(x):
        return criterion(model(x), y)
    backward(loss_fn, model.parameters(), x)
    optimizer.step()
```

## Installation

```bash
pip install xoe
```

Requires a JAX backend: `jax[cpu]`, `jax[cuda]`, or `jax[tpu]`.

## Features

- **Tensor** -- JAX-backed array with PyTorch-style operators (+, -, *, /, @, **, .T, .reshape, .squeeze, .unsqueeze, reductions, exp/log/tanh, indexing)
- **Autograd** -- `backward(loss_fn, params, *args)` using JAX's `grad` under the hood
- **nn** -- Linear, Sequential, Dropout, LayerNorm, ReLU, Sigmoid, Softmax, GELU, MSELoss, CrossEntropyLoss
- **Optim** -- SGD, Adam, AdamW
- **Random** -- seed, randn, zeros, ones, next_key (global PRNG key management)
- **Module** -- parameters(), state_dict(), load_state_dict(), zero_grad()
- **Init** -- zeros\_, ones\_, constant\_, kaiming_uniform\_, xavier_uniform\_
- **JIT** -- `@xoe.jit` decorator for XLA compilation

## Docs

Full documentation at [guide](/docs/guide/getting-started) and [API reference](/docs/api/tensor).

## Roadmap

See [goal.md](goal.md) for the full roadmap. Current status: Phases 1 & 2 complete (core API, losses, layers, activations, init). Phase 3 in progress (safetensors, JIT, data pipeline).
