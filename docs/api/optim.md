# Optim

Optimizers update model parameters based on computed gradients.

## SGD

```python
SGD(params, lr=0.01)
```

Stochastic gradient descent.

```python
optimizer = SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()
backward(loss_fn, model.parameters(), x)
optimizer.step()
```

## Adam

```python
Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8)
```

Standard Adam optimizer.

| Argument | Default | Description |
|---|---|---|
| `lr` | `0.001` | Learning rate |
| `betas` | `(0.9, 0.999)` | Coefficients for running averages |
| `eps` | `1e-8` | Term for numerical stability |

## AdamW

```python
AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
```

Adam with decoupled weight decay.

| Argument | Default | Description |
|---|---|---|
| `lr` | `0.001` | Learning rate |
| `betas` | `(0.9, 0.999)` | Coefficients for running averages |
| `eps` | `1e-8` | Term for numerical stability |
| `weight_decay` | `0.01` | Weight decay coefficient |

## Common Methods

All optimizers share this interface:

```python
optimizer.step()        # update parameters from gradients
optimizer.zero_grad()   # clear all parameter gradients
```
