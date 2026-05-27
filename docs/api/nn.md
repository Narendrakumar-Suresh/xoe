# nn API

Neural network layers, activations, loss functions, and the Module base class.

---

## Module

The base class for all neural network modules. Provides parameter management, state dicts, and train/eval mode.

```python
from xoe.nn import Module

class MyModel(Module):
    def __init__(self):
        self.fc = Linear(4, 1)

    def forward(self, x):
        return self.fc(x)
```

### Methods

| Method | Description |
|---|---|
| `parameters()` | Return a **list of Tensor** objects — all trainable parameters recursively |
| `state_dict()` | Return a flat `dict[str, jax.Array]` mapping parameter names to values |
| `load_state_dict(state)` | Load parameters from a state dict. Returns a list of error strings (empty on success). |
| `train(mode=True)` | Set the module and all children to training mode |
| `eval()` | Set the module and all children to evaluation mode |

```python
params = model.parameters()         # list of Tensors
sd = model.state_dict()             # {"fc.W": array(...), "fc.b": array(...)}
errors = model.load_state_dict(sd)  # restore from state dict
model.train()                       # enable dropout, batchnorm training
model.eval()                        # disable dropout, use running stats
```

### Important: `zero_grad()` lives on the optimizer

In xoe, `zero_grad()` is a method on the **optimizer**, not on the module. The optimizer knows which parameters it manages and clears their `.grad` attributes:

```python
optimizer.zero_grad()   # ✅ correct
```

---

## Linear

```python
Linear(in_features, out_features, bias=True, dtype=None)
```

A fully connected (dense) layer with learnable weight `W` of shape `(in_features, out_features)` and optional bias `b` of shape `(out_features,)`.

Weights are initialized with uniform Kaiming initialization by default.

```python
layer = Linear(4, 16)       # 4 inputs → 16 outputs
out = layer(x)              # x @ W + b

layer_no_bias = Linear(4, 16, bias=False)
```

---

## Sequential

```python
Sequential(*layers)
```

Chains multiple modules together. Input is passed through each layer in order.

```python
model = Sequential(
    Linear(4, 16),
    GELU(),
    Linear(16, 1),
)

out = model(x)  # forwards through all layers
```

You can index or iterate:

```python
first_layer = model[0]    # Linear(4, 16)
n_layers = len(model)     # 3
```

---

## Embedding

```python
Embedding(num_embeddings, embedding_dim, dtype=None, padding_idx=None, key=None)
```

A lookup table that maps integer indices to dense vectors.

| Argument | Default | Description |
|---|---|---|
| `num_embeddings` | required | Size of the dictionary (number of embeddings) |
| `embedding_dim` | required | Size of each embedding vector |
| `padding_idx` | `None` | If set, the embedding at this index is zeroed out |
| `dtype` | `None` | Data type |
| `key` | `None` | Optional PRNG key for initialization |

```python
embed = Embedding(100, 32)          # 100 tokens, 32-dim embeddings
indices = Tensor([1, 5, 20])        # integer indices
out = embed(indices)                # shape (3, 32)
```

---

## Dropout

```python
Dropout(p=0.5)
```

Randomly zeroes elements with probability `p` during training and scales remaining elements by `1 / (1 - p)`. During evaluation, acts as an identity function.

```python
dropout = Dropout(0.2)
out = dropout(x)        # drops 20% of elements during training
```

Dropout is only active when the module is in training mode. Call `model.eval()` to disable it during inference.

---

## LayerNorm

```python
LayerNorm(normalized_shape, eps=1e-5)
```

Normalizes the input across the last `normalized_shape` dimensions using learned affine parameters `weight` and `bias`. LayerNorm is applied as:

```
y = weight * (x - mean) / sqrt(var + eps) + bias
```

| Argument | Default | Description |
|---|---|---|
| `normalized_shape` | required | `int` or `tuple` — shape of the normalized dimensions (must be the trailing dimensions) |
| `eps` | `1e-5` | Small constant for numerical stability |

```python
ln = LayerNorm(16)
out = ln(x)             # normalizes last dimension

ln = LayerNorm((4, 16))
out = ln(x)             # normalizes last two dimensions
```

---

## BatchNorm1d

```python
BatchNorm1d(features, eps=1e-5, momentum=0.1, affine=True)
```

Batch Normalization for 2D inputs `(batch, features)`. Normalizes across the batch dimension and maintains running mean/variance for inference.

| Argument | Default | Description |
|---|---|---|
| `features` | required | Number of features |
| `eps` | `1e-5` | Numerical stability |
| `momentum` | `0.1` | Momentum for running statistics |
| `affine` | `True` | Learnable `weight` and `bias` |

```python
bn = BatchNorm1d(16)
out = bn(x)             # shape (batch, 16)
```

During training, running mean and variance are updated. During eval, the running statistics are used (call `model.eval()` before inference).

---

## Activations

All activations are available as both **functions** (in `xoe.nn.activations`) and **module classes**.

### Functions

```python
from xoe.nn.activations import relu, sigmoid, softmax, gelu, leaky_relu

relu(x)
sigmoid(x)
softmax(x, axis=-1)
gelu(x)
leaky_relu(x, negative_slope=0.01)
```

### Module Classes

| Class | Formula | Default params |
|---|---|---|
| `ReLU()` | `max(0, x)` | — |
| `Sigmoid()` | `1 / (1 + exp(-x))` | — |
| `Softmax(axis=-1)` | `exp(x) / sum(exp(x), axis)` | `axis=-1` |
| `GELU()` | `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` | — |
| `LeakyReLU(negative_slope=0.01)` | `max(negative_slope * x, x)` | `negative_slope=0.01` |

```python
from xoe.nn.activations import ReLU, Sigmoid, Softmax, GELU, LeakyReLU

relu = ReLU()
out = relu(x)
```

---

## Loss Functions

### MSELoss

```python
MSELoss(reduction="mean")
```

Mean Squared Error: `(pred - target)^2`

| Argument | Default | Description |
|---|---|---|
| `reduction` | `"mean"` | `"mean"`, `"sum"`, or `"none"` |

```python
criterion = MSELoss()
loss = criterion(pred, target)       # scalar Tensor
```

### CrossEntropyLoss

```python
CrossEntropyLoss(reduction="mean")
```

Cross-entropy loss with **log-softmax built in**. Accepts integer class indices or one-hot targets.

| Argument | Default | Description |
|---|---|---|
| `reduction` | `"mean"` | `"mean"`, `"sum"`, or `"none"` |

```python
criterion = CrossEntropyLoss()

# Integer targets (shape (batch,))
loss = criterion(logits, Tensor([0, 2, 1]))

# One-hot targets (shape (batch, classes))
loss = criterion(logits, one_hot_targets)
```
