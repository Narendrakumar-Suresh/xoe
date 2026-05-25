# Getting Started

## Installation

```bash
pip install xoe
```

Or with `uv`:

```bash
uv add xoe
```

xoe requires a JAX backend. Install the one that matches your hardware:

| Backend | Command |
|---|---|
| CPU | `pip install jax[cpu]` |
| CUDA | `pip install jax[cuda]` |
| TPU | `pip install jax[tpu]` |

## Quick Start

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

## Next Steps

- [Tensors guide](/guide/tensors) -- creation, operations, dtypes
- [Autograd guide](/guide/autograd) -- how backward() works
- [Training loop](/guide/training-loop) -- full training walkthrough
- [API reference](/api/tensor) -- Tensor, nn, optim, random
