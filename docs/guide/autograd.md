# Autograd

xoe uses JAX's `grad` under the hood. Gradients are computed by calling `backward()` with a **loss function**, not by calling `.backward()` on a tensor.

## Usage

```python
from xoe import Tensor, backward

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = Tensor([2.0, 4.0, 6.0], requires_grad=True)

def loss_fn(x, y):
    return ((x * y) ** 2).mean()

backward(loss_fn, [x, y], x, y)

print(x.grad)  # Tensor([...], requires_grad=False)
print(y.grad)  # Tensor([...], requires_grad=False)
```

## Why a function?

JAX requires pure functions for differentiation. `backward` hides this constraint from the user -- you pass a regular Python function and a list of parameters, and it handles the JAX transformation internally.

The loss function must:
- Accept the same arguments passed to `backward`
- Return a scalar `Tensor`

## Example with a loss module

```python
criterion = MSELoss()

def loss_fn(x):
    return criterion(model(x), y)

backward(loss_fn, model.parameters(), x)
```

After `backward()` completes, each parameter's `.grad` attribute is populated. Call `optimizer.step()` to update weights.
