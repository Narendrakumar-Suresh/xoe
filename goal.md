# Xoe Framework Goals

The goal is NOT to replace JAX.
The goal is to expose JAX power through a simpler, more intuitive, PyTorch-like interface while preserving functional correctness and compiler friendliness.

## Phase 1: Foundation & Core API (Complete)
- [x] **Core Tensor Class**: Thin jax.Array wrapper with ergonomic dunders, pytree registration
- [x] **Module System**: PyTorch-like Module with parameter discovery, state dicts
- [x] **Autograd Bridge**: Imperative wrapper using `jax.grad` — stores gradients on `param.grad`
- [x] **Basic Layers**: Linear layer
- [x] **Optimizers**: SGD, Adam, AdamW (stateful step/zero_grad API)
- [x] **State Dicts**: Flat-dict save/load with safetensors support
- [x] **JIT Compilation**: `@xoe.jit` decorator wrapping `jax.jit`

## Phase 2: Feature Parity (Complete)
- [x] **Loss Functions**: MSELoss, CrossEntropyLoss (log-sum-exp trick)
- [x] **Advanced Layers**: Sequential, Dropout, LayerNorm, BatchNorm1d
- [x] **Activations**: ReLU, Sigmoid, Softmax, GELU, LeakyReLU
- [x] **Initialization**: zeros, ones, constant, Kaiming, Xavier
- [x] **Embedding**: Embedding layer with padding_idx support

## Phase 3: Performance & Serialization (Partial)
- [x] **Safetensors Integration**: save_as_safetensor / load_from_safetensor
- [x] **JIT Compilation**: @xoe.jit for module-level JIT
- [x] **Automatic Batching**: vmap support across Modules

## Phase 4: Scale & Ecosystem (Not Started)
- [ ] **Distributed Training**: pmap support
- [ ] **Model Zoo**: Reference Transformer (GPT), ResNet
- [ ] **Benchmarking**: Performance comparison vs PyTorch/Equinox

---

*Last Updated: 2026-05-26*
