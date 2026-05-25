# Training Loop

A complete training loop in xoe follows the same pattern as PyTorch.

## Step-by-step

```python
import xoe
from xoe import Tensor, backward
from xoe.nn import Linear, Sequential
from xoe.nn.activations import GELU
from xoe.nn.functional import MSELoss
from xoe.optim import AdamW

model = Sequential(
    Linear(4, 16),
    GELU(),
    Linear(16, 1),
)

criterion = MSELoss()
optimizer = AdamW(model.parameters(), lr=1e-3)

x = Tensor([[1.0, 2.0, 3.0, 4.0]] * 8)
y = Tensor([[1.0]] * 8)

for step in range(20):
    optimizer.zero_grad()

    def loss_fn(x):
        return criterion(model(x), y)

    backward(loss_fn, model.parameters(), x)
    optimizer.step()
```

## The zero_grad pattern

Always call `zero_grad()` before computing gradients. Without it, the old gradients remain from the previous step.

```python
optimizer.zero_grad()   # clear old gradients

def loss_fn(x):
    return criterion(model(x), y)

backward(loss_fn, model.parameters(), x)  # compute new gradients
optimizer.step()                          # update weights
```

Alternatively, call `model.zero_grad()` if you prefer:

```python
model.zero_grad()
```
