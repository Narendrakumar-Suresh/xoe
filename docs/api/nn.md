# nn

Neural network layers, activations, and loss functions.

## Linear

```python
Linear(in_features, out_features, bias=True, dtype=None)
```

A fully connected layer with learnable weight `W` and optional bias `b`.

```python
layer = Linear(4, 16)
out = layer(x)  # x @ W + b
```

## Sequential

```python
Sequential(*layers)
```

Chains multiple modules. Forwards input through each layer in order.

```python
model = Sequential(
    Linear(4, 16),
    GELU(),
    Linear(16, 1),
)
```

## Dropout

```python
Dropout(p=0.5)
```

Randomly zeroes elements with probability `p` during training. Scales remaining elements by `1 / (1 - p)`.

```python
dropout = Dropout(0.2)
out = dropout(x)
```

## LayerNorm

```python
LayerNorm(normalized_shape, eps=1e-5)
```

Normalizes across the last `normalized_shape` dimensions.

```python
ln = LayerNorm(16)
out = ln(x)
```

## Activations

| Class | Function |
|---|---|
| `ReLU()` | `max(0, x)` |
| `Sigmoid()` | `1 / (1 + exp(-x))` |
| `Softmax(axis=-1)` | `exp(x) / sum(exp(x), axis)` |
| `GELU()` | `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` |

```python
from xoe.nn.activations import ReLU, Sigmoid, Softmax, GELU

relu = ReLU()
out = relu(x)
```

## Loss Functions

### MSELoss

```python
MSELoss(reduction="mean")
```

Mean squared error. `reduction` can be `"mean"`, `"sum"`, or `"none"`.

```python
criterion = MSELoss()
loss = criterion(pred, target)
```

### CrossEntropyLoss

```python
CrossEntropyLoss(reduction="mean")
```

Cross-entropy loss with log-softmax built in. Accepts integer or one-hot targets.

```python
criterion = CrossEntropyLoss()
loss = criterion(logits, targets)
```

## Module

Base class for all modules.

```python
module.parameters()     # list of trainable Tensors
module.state_dict()     # dict of name -> array
module.load_state_dict(state)
module.zero_grad()      # clear all parameter gradients
```
