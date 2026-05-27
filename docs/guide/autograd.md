# Autograd

xoe computes gradients using JAX's automatic differentiation under the hood. The entry point is the `backward()` function.

Unlike PyTorch where you call `loss.backward()` on a tensor, in xoe you pass a **loss function** to `backward()`. This is because JAX requires pure functions for differentiation, and `backward()` hides that requirement from you.

---

## How `backward()` Works

```python
from xoe import Tensor, backward

# Create some parameters
w = Tensor([[2.0]])
b = Tensor([1.0])
params = [w, b]

# Define a loss function that uses the parameters
def loss_fn(x):
    return (w * x + b).sum()

# Compute gradients — this sets w.grad and b.grad
backward(loss_fn, params, Tensor([3.0]))

# Now each parameter has its gradient stored in .grad
print(w.grad)   # Tensor containing d(loss)/dw
print(b.grad)   # Tensor containing d(loss)/db
```

### What happens inside `backward()`:

1. It saves the original data of each parameter
2. It constructs a pure function that JAX can differentiate
3. It calls `jax.grad` on that pure function
4. It restores each parameter's original data
5. It wraps each gradient as a `Tensor` and stores it in `param.grad`

### Important: `backward()` does NOT return anything

The old API used to return `(loss, grads)`. The new API stores gradients on the parameters themselves and returns nothing:

```python
# ❌ Old style (no longer works)
loss, grads = backward(loss_fn, params, x)
optimizer.step(grads)

# ✅ New style (store gradients on params)
backward(loss_fn, params, x)
optimizer.step()
```

To see the loss value, call your loss function directly:

```python
backward(loss_fn, params, x)
print(loss_fn(x))   # re-run to see current loss
```

---

## The Loss Function

Your loss function is a regular Python function that:

1. **Accepts the same extra arguments** that you pass to `backward()` after `params`
2. **Returns a scalar `Tensor`** (a 0-d tensor)

```python
def my_loss_fn(x, y):
    pred = model(x)
    return ((pred - y) ** 2).mean()

backward(my_loss_fn, model.parameters(), x, y)
```

The extra arguments (`x`, `y` in this example) are passed through unchanged to your loss function. Only the `params` list is traced by JAX for differentiation.

---

## Example with a Model

```python
from xoe.nn import Linear
from xoe.nn.functional import MSELoss

model = Linear(4, 1)
criterion = MSELoss()
params = model.parameters()
optimizer = SGD(params, lr=0.01)

x = Tensor([[1.0, 2.0, 3.0, 4.0]])
y = Tensor([[1.0]])

def loss_fn(x):
    return criterion(model(x), y)

# Forward + backward in one call
backward(loss_fn, params, x)

# Each parameter now has gradients
for p in params:
    print(f"param shape={p.shape}, grad shape={p.grad.shape}")

# Update weights
optimizer.step()
```

---

## Rules for the Loss Function

| Rule | Why |
|---|---|
| Must return a **scalar** Tensor | Gradients are only defined for scalar outputs |
| Must be a **pure function** of the parameters | JAX differentiates the computation graph, which must be side-effect free |
| Can use any xeo operations inside | All Tensor operations are traced by JAX automatically |
| Parameters are passed as a **list** | Each element in the list gets its own gradient |
