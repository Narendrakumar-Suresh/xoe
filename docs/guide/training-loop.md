# Training Loop

A complete training loop in xoe follows the same structure as PyTorch. If you've trained a model in PyTorch, this will look very familiar — the key difference is how gradients are computed.

---

## The Standard Pattern

Every training step has four phases:

```
1. zero_grad()   — clear old gradients from the previous step
2. backward()    — compute new gradients and store them on each parameter
3. step()        — update each parameter using its gradient
4. (optional)    — compute and print the current loss
```

Here's the complete code:

```python
import xoe
from xoe import Tensor, backward
from xoe.nn import Linear, Sequential
from xoe.nn.activations import GELU
from xoe.nn.functional import MSELoss
from xoe.optim import AdamW

# --- Build model ---
model = Sequential(
    Linear(4, 16),
    GELU(),
    Linear(16, 1),
)

# --- Data ---
x = Tensor([[1.0, 2.0, 3.0, 4.0]] * 8)
y = Tensor([[1.0]] * 8)

# --- Loss & optimizer ---
criterion = MSELoss()
params = model.parameters()
optimizer = AdamW(params, lr=1e-3)

# --- Training ---
for step in range(20):
    # 1. Clear old gradients
    optimizer.zero_grad()

    # 2. Define the loss function for this step
    def loss_fn(x):
        return criterion(model(x), y)

    # 3. Compute gradients
    backward(loss_fn, params, x)

    # 4. Update weights
    optimizer.step()

    # Print loss (re-run the model to get the value)
    print(f"step {step+1:02d} | loss: {loss_fn(x)}")
```

---

## Step-by-Step Breakdown

### Step 1: `optimizer.zero_grad()`

Clears the `.grad` attribute on every parameter the optimizer manages. This is essential — without it, gradients from the previous step would accumulate.

```python
optimizer.zero_grad()
# After this: all params have .grad = None
```

### Step 2: Define the loss function

You define a function that computes the loss given your model's input. This function is passed to `backward()` so JAX can differentiate it.

```python
def loss_fn(x):
    return criterion(model(x), y)
```

The function **must return a scalar Tensor**. It can use any xeo operations, call your model, apply loss functions, etc.

### Step 3: `backward(loss_fn, params, x)`

This is the core autograd call. It:

1. Takes your loss function and the list of parameters
2. Uses JAX's `grad` to compute gradients of the loss with respect to each parameter
3. Stores each gradient as a `Tensor` in `param.grad`
4. Restores the original parameter values (they get temporarily modified during JAX tracing)

After this call, every parameter in the list has its `.grad` populated:

```python
for p in params:
    print(p.grad)   # Tensor holding the gradient
```

### Step 4: `optimizer.step()`

Reads `p.grad` from each parameter, applies the optimization rule (SGD, Adam, AdamW, etc.), and updates `p.data` in place. Takes no arguments — it knows which parameters to update because you passed them to the constructor.

```python
optimizer.step()
# After this: each param's .data has been updated
```

---

## Computing the Loss for Printing

Since `backward()` no longer returns the loss value, you need to call your loss function again to get the current loss:

```python
backward(loss_fn, params, x)     # computes gradients
optimizer.step()                  # updates weights
current_loss = loss_fn(x)         # re-run to get the value
print(f"loss: {current_loss}")
```

Alternatively, call the loss function before `backward()`:

```python
current_loss = loss_fn(x)         # get loss BEFORE updating
backward(loss_fn, params, x)
optimizer.step()
print(f"loss: {current_loss}")
```

Both approaches are fine. The first approach prints the **post-update** loss (more common in practice).

---

## Full Example with Custom Module

```python
import xoe
from xoe import Tensor, backward
import xoe.nn as nn
from xoe.nn.functional import MSELoss
from xoe.optim import SGD

class MyModel(nn.Module):
    def __init__(self):
        self.l1 = nn.Linear(2, 8)
        self.l2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))

model = MyModel()
params = model.parameters()
optimizer = SGD(params, lr=0.01)

x = Tensor([[1.0, 2.0], [3.0, 4.0]])
y = Tensor([[1.0], [2.0]])

for step in range(10):
    optimizer.zero_grad()

    def loss_fn(x):
        return MSELoss()(model(x), y)

    backward(loss_fn, params, x)
    optimizer.step()
    print(f"step {step+1:02d} | loss: {loss_fn(x)}")
```
