# Xoe Framework Goals 🚀

This document tracks the long-term roadmap for the **xoe** deep learning framework. The goal is to create a "sweet spot" between PyTorch and JAX—imperative API, JAX performance, and simple state management.

## 🟢 Phase 1: Foundation & Core API (Current)
- [x] **Core Tensor Class**: Basic dunders and JAX integration.
- [x] **Autograd Bridge**: Imperative wrapper for `jax.grad`.
- [x] **Base Module**: PyTorch-like `Module` with parameter discovery.
- [x] **Basic Layers**: `Linear` layer implementation.
- [x] **Optimizers**: SGD, Adam, and AdamW.
- [x] **Global State**: Dynamic `set_default_dtype` and global precision management.
- [x] **State Dicts**: Flat-dict `state_dict()` and `load_state_dict()` for `safetensors` compatibility.

## 🟡 Phase 2: Feature Parity (Next Steps)
- [ ] **Loss Functions**: 
    - `MSELoss`
    - `CrossEntropyLoss` (with log-sum-exp trick for stability)
- [ ] **Advanced Layers**:
    - `Sequential` container
    - `Dropout`
    - `LayerNorm` / `BatchNorm1d`
- [ ] **Activations**: Add `GELU`, `LeakyReLU`, and `Softmax`.
- [ ] **Initialization**: Kaiming (He) and Xavier (Glorot) init strategies.

## 🟠 Phase 3: Performance & Serialization
- [ ] **Safetensors Integration**: Official `save` and `load` utilities.
- [ ] **JIT Compilation**: Implement Pytree registration or a JIT-wrapper for Modules.
- [ ] **Automatic Batching**: Support for `jax.vmap` across Modules.
- [ ] **Data Pipeline**: `Dataset` and `DataLoader` classes.

## 🔴 Phase 4: Scale & Ecosystem (Long-term)
- [ ] **Distributed Training**: Support for `jax.pmap` (Multi-GPU).
- [ ] **Model Zoo**: Implement a reference Transformer (GPT) and ResNet in xoe.
- [ ] **Documentation**: API Reference and Tutorials.
- [ ] **Benchmarking**: Performance comparison against PyTorch and Equinox.

---
*Last Updated: 2026-05-20*
