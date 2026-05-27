# Optim API

Optimizers update model parameters based on gradients. In xoe, optimizers follow a simple stateful pattern:

```python
optimizer = SGD(model.parameters(), lr=0.01)

optimizer.zero_grad()          # clear old gradients
backward(loss_fn, params, x)   # compute new gradients
optimizer.step()               # update parameters (no arguments needed)
```

---

## SGD

```python
SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False)
```

Stochastic Gradient Descent with optional momentum and weight decay.

| Argument | Default | Description |
|---|---|---|
| `lr` | `0.01` | Learning rate |
| `momentum` | `0.0` | Momentum factor. 0 = no momentum. |
| `weight_decay` | `0.0` | L2 penalty on parameters. Applied as `g += weight_decay * param`. |
| `nesterov` | `False` | Enable Nesterov accelerated gradient (only used when `momentum > 0`). |

```python
from xoe.optim import SGD

optimizer = SGD(model.parameters(), lr=0.01)
optimizer = SGD(params, lr=0.01, momentum=0.9)
optimizer = SGD(params, lr=0.01, weight_decay=1e-4)
optimizer = SGD(params, lr=0.01, momentum=0.9, nesterov=True)
```

---

## Adam

```python
Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
```

Adaptive Moment Estimation — combines momentum with per-parameter learning rates.

| Argument | Default | Description |
|---|---|---|
| `lr` | `0.001` | Learning rate |
| `betas` | `(0.9, 0.999)` | Coefficients for running averages of gradient and squared gradient |
| `eps` | `1e-8` | Small constant for numerical stability |
| `amsgrad` | `False` | Use AMSGrad variant (uses maximum of past squared gradients) |

```python
from xoe.optim import Adam

optimizer = Adam(model.parameters())
optimizer = Adam(params, lr=0.001, betas=(0.9, 0.999))
optimizer = Adam(params, lr=0.001, amsgrad=True)
```

---

## AdamW

```python
AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)
```

Adam with **decoupled weight decay**. Unlike L2 regularization in SGD, the weight decay is applied directly to the parameter values **after** the Adam update, which often leads to better generalization.

| Argument | Default | Description |
|---|---|---|
| `lr` | `0.001` | Learning rate |
| `betas` | `(0.9, 0.999)` | Coefficients for running averages |
| `eps` | `1e-8` | Numerical stability |
| `weight_decay` | `0.01` | Decoupled weight decay coefficient |
| `amsgrad` | `False` | Use AMSGrad variant |

```python
from xoe.optim import AdamW

optimizer = AdamW(model.parameters())
optimizer = AdamW(params, lr=1e-3, weight_decay=0.01)
```

---

## Common Methods

All optimizers share the same interface:

### `.step()`

Updates all parameters using their current gradients. Takes **no arguments**. Reads `p.grad.data` from each stored parameter and updates `p.data` in place.

```python
optimizer.step()
```

### `.zero_grad()`

Sets `p.grad = None` on every parameter the optimizer manages. Call this at the **start** of each training step to clear gradients from the previous iteration.

```python
optimizer.zero_grad()
```

---

## Training Loop Pattern

The standard pattern for using any optimizer:

```python
optimizer = AdamW(model.parameters(), lr=1e-3)

for step in range(num_steps):
    optimizer.zero_grad()

    def loss_fn(x):
        return criterion(model(x), y)

    backward(loss_fn, model.parameters(), x)
    optimizer.step()
```
