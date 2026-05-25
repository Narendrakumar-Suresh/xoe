You are writing documentation for **xoe**, a deep learning framework.
Tagline: "Write PyTorch. Run JAX."
Description: "A deep learning framework with PyTorch-style imperative API backed by JAX and XLA. Stateful, fast, and built for TPUs."

The docs use **VitePress**. Write clean, minimal markdown.

## Your task
Write the following VitePress docs pages:

### `docs/index.md` — landing page
Hero section with tagline, description, and two CTA buttons: "Get Started" and "GitHub".
Feature cards: "PyTorch Ergonomics", "JAX Performance", "TPU Native", "Zero Boilerplate".

### `docs/guide/getting-started.md`
- Installation via `pip install xoe` and `uv add xoe`
- Quick start: a simple MLP training loop using xoe (use the example below)
- Mention jax[cpu] / jax[cuda] / jax[tpu] backend options

### `docs/guide/tensors.md`
- Tensor creation
- All ops: +, -, *, /, @, **, .T, .reshape, .squeeze, .unsqueeze
- Reductions: .sum, .mean, .max
- dtype support: float32, float16, bfloat16
- set_default_dtype

### `docs/guide/autograd.md`
- How backward() works
- Why loss must be a function (pure fn requirement, JAX constraint, hidden from user)
- Example with a real loss

### `docs/guide/training-loop.md`
- Full training loop: model, loss, optimizer, backward, step
- zero_grad pattern

### `docs/api/tensor.md`
- Document every method on Tensor class

### `docs/api/nn.md`
- Linear, Sequential, Dropout, LayerNorm, ReLU, Sigmoid, Softmax, GELU, MSELoss, CrossEntropyLoss

### `docs/api/optim.md`
- SGD, Adam, AdamW with all args

### `docs/api/random.md`
- seed, randn, zeros, ones, next_key

### `docs/.vitepress/config.ts`
- title: "xoe"
- description: "Write PyTorch. Run JAX."
- nav: Guide, API, GitHub
- sidebar for guide/ and api/
- dark mode default

## Reference code (use this for examples):

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

## Style rules
- No fluff, no filler sentences
- Short paragraphs
- Code examples for everything
- No emojis except on landing page feature cards