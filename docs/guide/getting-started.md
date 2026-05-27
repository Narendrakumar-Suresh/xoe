# Getting Started

Welcome to **xoe** — a deep learning framework that lets you **write PyTorch-style code and run it on JAX**.

If you already know PyTorch, you already know xoe. The same imperative feel, the same training loop pattern, the same module system — but backed by JAX's XLA compilation, automatic differentiation, and first-class TPU support.

---

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

---

## Quick Start

Here's a complete training loop that builds a neural network, generates synthetic data, and trains for 20 steps:

```python
import xoe
from xoe import Tensor, backward
from xoe.nn import Linear, Sequential, LayerNorm
from xoe.nn.activations import GELU
from xoe.nn.functional import MSELoss
from xoe.optim import AdamW

# --- Build the model ---
model = Sequential(
    Linear(4, 16),     # 4 inputs -> 16 hidden
    GELU(),            # activation
    LayerNorm(16),     # normalize hidden layer
    Linear(16, 1),     # 16 hidden -> 1 output
)

# --- Synthetic data ---
x = Tensor([[1.0, 2.0, 3.0, 4.0]] * 8)
y = Tensor([[1.0]] * 8)

# --- Loss and optimizer ---
criterion = MSELoss()
optimizer = AdamW(model.parameters(), lr=1e-3)

# --- Training loop ---
for step in range(20):
    optimizer.zero_grad()

    def loss_fn(x):
        return criterion(model(x), y)

    backward(loss_fn, model.parameters(), x)
    optimizer.step()

    # Re-run the model to get the current loss for printing
    print(f"step {step+1:02d} | loss: {loss_fn(x)}")
```

### What's happening step by step:

| Line | What it does |
|---|---|
| `optimizer.zero_grad()` | Clears old gradients from previous step |
| `def loss_fn(x): ...` | Defines the function we want to differentiate |
| `backward(loss_fn, params, x)` | Computes gradients and stores them on each parameter's `.grad` attribute |
| `optimizer.step()` | Reads `.grad` from each parameter and updates its `.data` in place |

---

## Next Steps

| Guide | What you'll learn |
|---|---|
| [Tensors](/guide/tensors) | Creating tensors, operations, shapes, dtypes |
| [Autograd](/guide/autograd) | How `backward()` computes gradients |
| [Training Loop](/guide/training-loop) | Full walkthrough of training patterns |
| [API Reference](/api/tensor) | Complete docs for Tensor, nn, optim, random |
