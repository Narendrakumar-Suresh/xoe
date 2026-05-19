myfw/
├── myfw/
│   ├── tensor.py
│   ├── nn/
│   ├── optim/
│   ├── autograd.py
│   └── random.py
├── tests/
├── pyproject.toml
└── README.md


**build order:**

**1. `tensor.py`** — foundation, everything depends on it
- all ops (`+`, `-`, `*`, `/`, `@`, `**`)
- shape ops (`.T`, `.reshape`, `.squeeze`, `.unsqueeze`)
- reduction ops (`.sum`, `.mean`, `.max`)
- comparison dunders
- `__repr__`

**2. `random.py`** — needed by nn layers for weight init
- global PRNG state
- `seed()`, `next_key()`
- `randn()`, `zeros()`, `ones()`

**3. `autograd.py`** — needed by optimizer
- `backward(loss_fn, params)`
- grad accumulation into `tensor.grad`

**4. `nn/module.py`** — base class
- `__call__` → `forward`
- `parameters()` — recurse through `__dict__`
- `zero_grad()`

**5. `nn/linear.py`** — first real layer, proves module works
- `__init__` uses `random.py`
- `forward` uses tensor ops

**6. `optim/`** — SGD first, Adam second
- `step()` updates params using `.grad`
- `zero_grad()` delegates to module

**7. `nn/activations.py`** — relu, sigmoid, softmax, gelu

test after each step against numpy/PyTorch ground truth before moving on.