import jax
import jax.numpy as jnp
import pytest
import numpy as np

import xoe
from xoe import Tensor, zeros, ones, backward
import xoe.nn as nn
from xoe.nn.activations import relu, sigmoid, softmax, gelu
from xoe.nn.functional import MSELoss, CrossEntropyLoss
from xoe.nn import init
from xoe.optim import SGD, Adam, AdamW
from xoe import store


# ── Tensor ──────────────────────────────────────────────────────────────


class TestTensor:
    def test_create_from_list(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert isinstance(t, Tensor)
        assert t.shape == (3,)
        assert t.dtype == jnp.float32

    def test_create_from_jax_array(self):
        a = jnp.array([1.0, 2.0, 3.0])
        t = Tensor(a)
        assert t.data is a  # zero-copy

    def test_create_from_tensor(self):
        t1 = Tensor([1.0, 2.0])
        t2 = Tensor(t1)
        assert isinstance(t2, Tensor)
        assert t2.shape == (2,)

    def test_create_with_dtype(self):
        t = Tensor([1, 2, 3], dtype=jnp.float16)
        assert t.dtype == jnp.float16

    def test_dtype_property(self):
        t = Tensor([1.0])
        assert t.dtype == jnp.float32

    def test_shape_property(self):
        t = Tensor([[1, 2], [3, 4]])
        assert t.shape == (2, 2)

    def test_ndim_property(self):
        t = Tensor([[1, 2], [3, 4]])
        assert t.ndim == 2

    def test_T_property(self):
        t = Tensor([[1, 2], [3, 4]])
        assert t.T.shape == (2, 2)

    def test_data_getter_setter(self):
        t = Tensor([1.0, 2.0])
        assert isinstance(t.data, jnp.ndarray)
        t.data = jnp.array([3.0, 4.0])
        assert t.data.tolist() == [3.0, 4.0]

    def test_numpy(self):
        t = Tensor([1.0, 2.0])
        a = t.numpy()
        assert isinstance(a, np.ndarray)

    def test_item(self):
        t = Tensor([42.0])
        assert t.item() == 42.0

    def test_tolist(self):
        t = Tensor([1.0, 2.0])
        assert t.tolist() == [1.0, 2.0]

    def test_reshape(self):
        t = Tensor([1, 2, 3, 4])
        r = t.reshape(2, 2)
        assert r.shape == (2, 2)

    def test_squeeze(self):
        t = Tensor([[1], [2], [3]])
        s = t.squeeze()
        assert s.shape == (3,)

    def test_unsqueeze(self):
        t = Tensor([1, 2, 3])
        u = t.unsqueeze(0)
        assert u.shape == (1, 3)

    def test_detach(self):
        t = Tensor([1.0, 2.0])
        d = t.detach()
        assert d.data.tolist() == [1.0, 2.0]

    def test_sum(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.sum().item() == 6.0

    def test_mean(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.mean().item() == 2.0

    def test_max_min(self):
        t = Tensor([3.0, 1.0, 2.0])
        assert t.max().item() == 3.0
        assert t.min().item() == 1.0

    def test_getitem(self):
        t = Tensor([10, 20, 30])
        assert t[1].item() == 20

    def test_len(self):
        t = Tensor([1, 2, 3])
        assert len(t) == 3

    def test_exp_log_tanh(self):
        t = Tensor([1.0])
        assert abs(t.exp().item() - jnp.e) < 1e-5
        assert abs(t.log().item()) < 1e-5
        assert abs(t.tanh().item() - jnp.tanh(1.0)) < 1e-5

    def test_arithmetic(self):
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        assert (a + b).tolist() == [4.0, 6.0]
        assert (b - a).tolist() == [2.0, 2.0]
        assert (a * b).tolist() == [3.0, 8.0]
        assert (b / a).tolist() == [3.0, 2.0]

    def test_matmul(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0], [6.0]])
        c = a @ b
        assert c.shape == (2, 1)

    def test_pow_neg(self):
        t = Tensor([2.0, 3.0])
        assert (t**2).tolist() == [4.0, 9.0]
        assert (-t).tolist() == [-2.0, -3.0]

    def test_comparisons(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([1.0, 5.0, 3.0])
        assert (a == b).tolist() == [True, False, True]
        assert (a != b).tolist() == [False, True, False]
        assert (a < b).tolist() == [False, True, False]

    def test_bool_error(self):
        t = Tensor([1, 2])
        with pytest.raises(ValueError, match="ambiguous"):
            bool(t)

    def test_repr(self):
        t = Tensor([1.0])
        assert "Tensor" in repr(t)

    def test_pytree_flatten_unflatten(self):
        t = Tensor([1.0, 2.0, 3.0])
        leaves, treedef = jax.tree_util.tree_flatten(t)
        assert len(leaves) == 1
        assert isinstance(leaves[0], jnp.ndarray)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(restored, Tensor)
        assert restored.tolist() == [1.0, 2.0, 3.0]

    def test_zeros_ones(self):
        z = zeros((2, 3))
        assert z.shape == (2, 3)
        assert jnp.all(z == 0)
        o = ones((2, 3))
        assert o.shape == (2, 3)
        assert jnp.all(o == 1)

    def test_with_scalar(self):
        t = Tensor(42.0)
        assert t.shape == ()
        assert t.item() == 42.0

    def test_broadcasting(self):
        a = Tensor([[1.0, 2.0, 3.0]])
        b = Tensor([[1.0], [2.0], [3.0]])
        c = a + b
        assert c.shape == (3, 3)

    def test_radd_rmul(self):
        t = Tensor([1.0, 2.0])
        assert (1.0 + t).tolist() == [2.0, 3.0]
        assert (2 * t).tolist() == [2.0, 4.0]

    def test_rsub_rtruediv(self):
        t = Tensor([1.0, 2.0])
        assert (10.0 - t).tolist() == [9.0, 8.0]
        assert (10.0 / t).tolist() == [10.0, 5.0]

    def test_rmatmul(self):
        a = Tensor([[1.0, 2.0]])
        b = jnp.array([[3.0], [4.0]])
        c = b @ a
        assert c.shape == (2, 2)


# ── Autograd ────────────────────────────────────────────────────────────


class TestAutograd:
    def test_backward_simple(self):
        w = Tensor([[2.0]])
        b = Tensor([1.0])
        params = [w, b]

        def loss_fn(x):
            return (w * x + b).sum()

        backward(loss_fn, params, Tensor([3.0]))
        assert isinstance(w.grad, Tensor)
        assert w.grad.shape == (1, 1)
        assert isinstance(b.grad, Tensor)
        assert b.grad.shape == (1,)

    def test_backward_multiple_params(self):
        w1 = Tensor([[1.0, 2.0], [3.0, 4.0]])
        w2 = Tensor([[2.0], [-1.0]])
        b1 = Tensor([0.1, 0.2])
        b2 = Tensor([0.5])
        params = [w1, b1, w2, b2]

        def loss_fn(x):
            h = x @ w1 + b1
            a = relu(h)
            y = a @ w2 + b2
            return y.sum()

        x = Tensor([[1.0, 2.0]])
        backward(loss_fn, params, x)
        for p in params:
            assert isinstance(p.grad, Tensor)

    def test_backward_no_tensor_result(self):
        w = Tensor([[1.0]])
        params = [w]

        def loss_fn(x):
            return float((w * x).sum().item())

        backward(loss_fn, params, Tensor([2.0]))
        assert isinstance(w.grad, Tensor)

    def test_backward_preserves_params(self):
        w = Tensor([[1.0]])
        orig = w.data.copy()
        params = [w]

        def loss_fn(x):
            return (w * x).sum()

        backward(loss_fn, params, Tensor([3.0]))
        assert jnp.all(w.data == orig)


# ── Module ──────────────────────────────────────────────────────────────


class TestModule:
    def test_module_parameters(self):
        model = nn.Linear(4, 8)
        params = model.parameters()
        assert len(params) == 2  # W and b
        assert all(isinstance(p, Tensor) for p in params)

    def test_module_parameters_seq(self):
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        params = model.parameters()
        assert len(params) == 4  # 2 Linear layers × 2 params each

    def test_state_dict_roundtrip(self):
        model = nn.Linear(4, 8)
        state = model.state_dict()
        assert "W" in state
        assert "b" in state

        new_model = nn.Linear(4, 8)
        errors = new_model.load_state_dict(state)
        assert errors == []
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert jnp.all(p1.data == p2.data)

    def test_load_state_dict_shape_mismatch(self):
        model = nn.Linear(4, 8)
        state = {"W": jnp.zeros((4, 16))}
        errors = model.load_state_dict(state)
        assert any("Shape mismatch" in e for e in errors)

    def test_load_state_dict_missing_key(self):
        model = nn.Linear(4, 8)
        bad_state = {"nonexistent": jnp.zeros((4, 8))}
        errors = model.load_state_dict(bad_state)
        assert any("Missing key" in e for e in errors)

    def test_train_eval(self):
        model = nn.Sequential(nn.Dropout(0.5))
        model.train()
        assert model.training is True
        assert model.layers[0].training is True
        model.eval()
        assert model.training is False
        assert model.layers[0].training is False

    def test_forward(self):
        model = nn.Linear(4, 3)
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        out = model(x)
        assert out.shape == (1, 3)

    def test_sequential_forward(self):
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        out = model(x)
        assert out.shape == (1, 2)

    def test_sequential_getitem(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        assert isinstance(model[0], nn.Linear)
        assert isinstance(model[1], nn.ReLU)
        assert len(model) == 2


# ── Layers ──────────────────────────────────────────────────────────────


class TestLinear:
    def test_forward(self):
        layer = nn.Linear(4, 3)
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        out = layer(x)
        assert out.shape == (1, 3)

    def test_no_bias(self):
        layer = nn.Linear(4, 3, bias=False)
        assert layer.b is None
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        out = layer(x)
        assert out.shape == (1, 3)

    def test_dtype(self):
        layer = nn.Linear(4, 3, dtype=jnp.float16)
        assert layer.W.dtype == jnp.float16


class TestEmbedding:
    def test_forward(self):
        emb = nn.Embedding(10, 8, key=jax.random.PRNGKey(0))
        idx = Tensor([1, 2, 3])
        out = emb(idx)
        assert out.shape == (3, 8)

    def test_padding_idx(self):
        emb = nn.Embedding(10, 4, padding_idx=0, key=jax.random.PRNGKey(0))
        idx = Tensor([0, 1, 2])
        out = emb(idx)
        assert out.shape == (3, 4)
        assert jnp.all(out.data[0] == 0)  # padding zeroed

    def test_raw_array_input(self):
        emb = nn.Embedding(10, 4, key=jax.random.PRNGKey(0))
        idx = jnp.array([1, 2, 3])
        out = emb(idx)
        assert isinstance(out, Tensor)


class TestLayerNorm:
    def test_forward(self):
        ln = nn.LayerNorm(4)
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        out = ln(x)
        assert out.shape == (1, 4)

    def test_int_shape(self):
        ln = nn.LayerNorm(4)
        assert ln.normalized_shape == (4,)


class TestDropout:
    def test_train_mode(self):
        drop = nn.Dropout(0.5)
        drop.train()
        x = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 16)
        out = drop(x)
        assert out.shape == (16, 6)
        # mean should be close to original mean (inverted scaling preserves expectation)
        assert abs(out.mean().item() - x.mean().item()) < 1.0

    def test_eval_mode(self):
        drop = nn.Dropout(0.5)
        drop.eval()
        x = Tensor([[1.0, 2.0, 3.0]])
        out = drop(x)
        assert out.tolist() == [[1.0, 2.0, 3.0]]

    def test_p_zero(self):
        drop = nn.Dropout(0.0)
        drop.train()
        x = Tensor([[1.0, 2.0]])
        out = drop(x)
        assert out.tolist() == [[1.0, 2.0]]


# ── Activations ─────────────────────────────────────────────────────────


class TestBatchNorm1d:
    def test_forward_train(self):
        bn = nn.BatchNorm1d(4)
        bn.train()
        x = Tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        out = bn(x)
        assert out.shape == (2, 4)
        assert abs(out.mean().item()) < 1e-5  # zero mean in training

    def test_forward_eval(self):
        bn = nn.BatchNorm1d(4)
        bn.eval()
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        out = bn(x)
        assert out.shape == (1, 4)

    def test_running_stats_update(self):
        bn = nn.BatchNorm1d(4, momentum=0.5)
        bn.train()
        x = Tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        bn(x)
        assert not jnp.all(bn._running_mean.data == 0)
        assert not jnp.all(bn._running_var.data == 1)

    def test_running_stats_used_in_eval(self):
        bn = nn.BatchNorm1d(4)
        bn.train()
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        bn(x)  # populate running stats
        bn.eval()
        out = bn(x)
        assert out.shape == (1, 4)

    def test_parameters_exclude_running_stats(self):
        bn = nn.BatchNorm1d(4)
        params = bn.parameters()
        names = [k for k in bn.__dict__ if isinstance(bn.__dict__[k], Tensor)]
        assert "_running_mean" in names
        assert all(not p is bn._running_mean for p in params)
        assert all(not p is bn._running_var for p in params)

    def test_no_affine(self):
        bn = nn.BatchNorm1d(4, affine=False)
        assert not hasattr(bn, "weight")
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        out = bn(x)
        assert out.shape == (1, 4)

    def test_state_dict_includes_running_stats(self):
        bn = nn.BatchNorm1d(4)
        state = bn.state_dict()
        assert "_running_mean" in state
        assert "_running_var" in state


class TestLeakyReLU:
    def test_functional(self):
        from xoe.nn.activations import leaky_relu

        x = Tensor([-1.0, 0.0, 1.0])
        out = leaky_relu(x)
        assert out[0].item() == pytest.approx(-0.01, abs=1e-6)
        assert out[1].item() == 0.0
        assert out[2].item() == 1.0

    def test_module(self):
        x = Tensor([-2.0, 0.0, 3.0])
        out = nn.LeakyReLU()(x)
        assert out[0].item() == pytest.approx(-0.02, abs=1e-6)
        assert out[1].item() == 0.0
        assert out[2].item() == 3.0

    def test_custom_slope(self):
        x = Tensor([-1.0, 0.0, 1.0])
        out = nn.LeakyReLU(negative_slope=0.1)(x)
        assert out[0].item() == pytest.approx(-0.1, abs=1e-6)
        assert out[1].item() == 0.0
        assert out[2].item() == 1.0


class TestActivations:
    def test_relu(self):
        x = Tensor([-1.0, 0.0, 1.0])
        assert relu(x).tolist() == [0.0, 0.0, 1.0]

    def test_sigmoid(self):
        x = Tensor([0.0])
        assert abs(sigmoid(x).item() - 0.5) < 1e-5

    def test_softmax(self):
        x = Tensor([[1.0, 2.0, 3.0]])
        out = softmax(x)
        assert abs(out.sum().item() - 1.0) < 1e-5

    def test_gelu(self):
        x = Tensor([0.0])
        assert abs(gelu(x).item()) < 1e-5

    def test_module_activations(self):
        x = Tensor([-1.0, 0.0, 1.0])
        assert nn.ReLU()(x).tolist() == [0.0, 0.0, 1.0]
        assert abs(nn.Sigmoid()(Tensor([0.0])).item() - 0.5) < 1e-5
        assert abs(nn.GELU()(Tensor([0.0])).item()) < 1e-5
        out = nn.Softmax()(Tensor([[1.0, 2.0]]))
        assert abs(out.sum().item() - 1.0) < 1e-5


# ── Loss Functions ──────────────────────────────────────────────────────


class TestLosses:
    def test_mse_loss_mean(self):
        pred = Tensor([1.0, 2.0, 3.0])
        target = Tensor([1.0, 2.0, 4.0])
        loss = MSELoss()(pred, target)
        assert loss.shape == ()
        assert abs(loss.item() - 1.0 / 3.0) < 1e-5

    def test_mse_loss_sum(self):
        pred = Tensor([1.0, 2.0])
        target = Tensor([1.0, 3.0])
        loss = MSELoss(reduction="sum")(pred, target)
        assert loss.item() == 1.0

    def test_mse_loss_none(self):
        pred = Tensor([1.0, 2.0, 3.0])
        target = Tensor([1.0, 2.0, 4.0])
        loss = MSELoss(reduction="none")(pred, target)
        assert loss.shape == (3,)

    def test_cross_entropy_integer_targets(self):
        logits = Tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        targets = Tensor([2, 1])
        loss = CrossEntropyLoss()(logits, targets)
        assert loss.shape == ()

    def test_cross_entropy_one_hot(self):
        logits = Tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        targets = Tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        loss = CrossEntropyLoss()(logits, targets)
        assert loss.shape == ()

    def test_cross_entropy_1d_no_batch(self):
        logits = Tensor([1.0, 2.0, 3.0])
        targets = Tensor([2])
        loss = CrossEntropyLoss()(logits, targets)
        assert loss.shape == ()

    def test_cross_entropy_sum(self):
        logits = Tensor([[1.0, 2.0, 3.0]])
        targets = Tensor([2])
        loss = CrossEntropyLoss(reduction="sum")(logits, targets)
        assert loss.shape == ()

    def test_cross_entropy_none(self):
        logits = Tensor([[1.0, 2.0, 3.0]])
        targets = Tensor([2])
        loss = CrossEntropyLoss(reduction="none")(logits, targets)
        assert loss.shape == (1,)


# ── Optimizers ──────────────────────────────────────────────────────────


class TestOptimizers:
    def test_sgd_step(self):
        w = Tensor([[1.0]])
        params = [w]
        opt = SGD(params, lr=0.1)
        init_w = w.data.copy()
        w.grad = Tensor._wrap(jnp.array([[2.0]]))
        opt.step()
        assert w.data[0, 0] < init_w[0, 0]  # weight decreased

    def test_sgd_momentum(self):
        w = Tensor([[1.0]])
        params = [w]
        opt = SGD(params, lr=0.1, momentum=0.9)
        w.grad = Tensor._wrap(jnp.array([[2.0]]))
        opt.step()
        init_w = w.data.copy()
        opt.step()
        assert w.data[0, 0] < init_w[0, 0]

    def test_sgd_weight_decay(self):
        w = Tensor([[1.0]])
        params = [w]
        opt = SGD(params, lr=0.01, weight_decay=0.1)
        init_w = w.data.copy()
        w.grad = Tensor._wrap(jnp.array([[0.0]]))
        opt.step()
        assert w.data[0, 0] < init_w[0, 0]  # weight decay reduces even with zero grad

    def test_adam_step(self):
        w = Tensor([[1.0]])
        params = [w]
        opt = Adam(params, lr=0.1)
        init_w = w.data.copy()
        w.grad = Tensor._wrap(jnp.array([[2.0]]))
        opt.step()
        assert w.data[0, 0] < init_w[0, 0]

    def test_adamw_step(self):
        w = Tensor([[1.0]])
        params = [w]
        opt = AdamW(params, lr=0.1)
        init_w = w.data.copy()
        w.grad = Tensor._wrap(jnp.array([[2.0]]))
        opt.step()
        assert w.data[0, 0] < init_w[0, 0]

    def test_adam_amsgrad(self):
        w = Tensor([[1.0]])
        params = [w]
        opt = Adam(params, lr=0.1, amsgrad=True)
        w.grad = Tensor._wrap(jnp.array([[2.0]]))
        opt.step()
        assert isinstance(w.data, jnp.ndarray)

    def test_optimizer_full_training_loop(self):
        model = nn.Linear(2, 1)
        params = model.parameters()
        opt = SGD(params, lr=0.01)
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y = Tensor([[1.0], [2.0]])

        def loss_fn(x):
            return MSELoss()(model(x), y)

        for _ in range(10):
            opt.zero_grad()
            backward(loss_fn, params, x)
            opt.step()

    def test_multiple_param_groups_inplace(self):
        w1 = Tensor([[1.0]])
        w2 = Tensor([[2.0]])
        params = [w1, w2]
        opt = Adam(params, lr=0.01)
        w1.grad = Tensor._wrap(jnp.array([[0.5]]))
        w2.grad = Tensor._wrap(jnp.array([[0.3]]))
        opt.step()
        assert w1.data[0, 0] < 1.0
        assert w2.data[0, 0] < 2.0


# ── Init ────────────────────────────────────────────────────────────────


class TestInit:
    def test_zeros_(self):
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        init.zeros_(t)
        assert jnp.all(t.data == 0)

    def test_ones_(self):
        t = Tensor([[0.0, 0.0]])
        init.ones_(t)
        assert jnp.all(t.data == 1)

    def test_constant_(self):
        t = Tensor([[0.0, 0.0]])
        init.constant_(t, 3.0)
        assert jnp.all(t.data == 3.0)

    def test_kaiming_uniform(self):
        t = Tensor(jnp.zeros((4, 8)))
        init.kaiming_uniform_(t, key=jax.random.PRNGKey(0))
        assert t.data.shape == (4, 8)
        assert not jnp.all(t.data == 0)

    def test_xavier_uniform(self):
        t = Tensor(jnp.zeros((4, 8)))
        init.xavier_uniform_(t, key=jax.random.PRNGKey(0))
        assert t.data.shape == (4, 8)
        assert not jnp.all(t.data == 0)


# ── JIT ─────────────────────────────────────────────────────────────────


class TestJIT:
    def test_jit_basic(self):
        @xoe.jit
        def f(x):
            return x * 2

        t = Tensor([1.0, 2.0, 3.0])
        result = f(t)
        assert isinstance(result, Tensor)
        assert result.tolist() == [2.0, 4.0, 6.0]

    def test_jit_multiple_args(self):
        @xoe.jit
        def f(a, b):
            return a + b

        x = Tensor([1.0, 2.0])
        y = Tensor([3.0, 4.0])
        result = f(x, y)
        assert result.tolist() == [4.0, 6.0]

    def test_jit_returns_tuple(self):
        @xoe.jit
        def f(x):
            return x, x * 2

        t = Tensor([1.0, 2.0])
        a, b = f(t)
        assert isinstance(a, Tensor)
        assert isinstance(b, Tensor)
        assert a.tolist() == [1.0, 2.0]
        assert b.tolist() == [2.0, 4.0]

    def test_jit_with_kwargs(self):
        @xoe.jit
        def f(x, scale=2.0):
            return x * scale

        t = Tensor([1.0, 2.0])
        result = f(t, scale=3.0)
        assert result.tolist() == [3.0, 6.0]

    def test_jit_caches(self):
        @xoe.jit
        def f(x):
            return x * 2

        t1 = Tensor([1.0, 2.0])
        t2 = Tensor([3.0, 4.0])
        r1 = f(t1)
        r2 = f(t2)
        assert r1.tolist() == [2.0, 4.0]
        assert r2.tolist() == [6.0, 8.0]


# ── Random ──────────────────────────────────────────────────────────────


class TestRandom:
    def test_seed(self):
        import xoe.random as rnd

        rnd.seed(42)
        k1 = rnd.next_key()
        rnd.seed(42)
        k2 = rnd.next_key()
        assert jnp.all(k1 == k2)

    def test_next_key_functional(self):
        import xoe.random as rnd

        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        result = rnd.next_key(key=k1)
        # next_key splits the input key and returns the second subkey
        _, expected = jax.random.split(k1)
        assert jnp.all(result == expected)

    def test_randn_shape(self):
        import xoe.random as rnd

        key = jax.random.PRNGKey(0)
        samples = rnd.randn((1000,), key=key)
        assert samples.shape == (1000,)
        assert abs(samples.mean()) < 0.2
        assert abs(samples.std() - 1.0) < 0.1

    def test_randn_dtype(self):
        import xoe.random as rnd

        key = jax.random.PRNGKey(0)
        samples = rnd.randn((4,), key=key, dtype=jnp.float16)
        assert samples.dtype == jnp.float16

    def test_split_key(self):
        import xoe.random as rnd

        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        k1b = rnd.split_key(key)
        # split_key returns new_keys[1], which is the second key from split
        assert jnp.all(k1b == k2)


# ── Store ───────────────────────────────────────────────────────────────


class TestStore:
    def test_save_load(self, tmp_path):
        model = nn.Linear(4, 8)
        path = str(tmp_path / "model.safetensors")
        store.save_as_safetensor(model, path)
        new_model = nn.Linear(4, 8)
        store.load_from_safetensor(new_model, path)
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert jnp.all(p1.data == p2.data)


# ── Integration ─────────────────────────────────────────────────────────


class TestIntegration:
    def test_end_to_end_training(self):
        model = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.LayerNorm(16),
            nn.Linear(16, 1),
        )
        x = Tensor([[1.0, 2.0, 3.0, 4.0]] * 8)
        y = Tensor([[1.0]] * 8)
        criterion = MSELoss()
        params = model.parameters()
        optimizer = AdamW(params, lr=1e-3)

        def loss_fn(x):
            return criterion(model(x), y)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            backward(loss_fn, params, x)
            optimizer.step()
            losses.append(loss_fn(x).item())
        assert losses[-1] < losses[0]

    def test_grad_flow_through_all_layers(self):
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        params = model.parameters()

        def loss_fn(x):
            return model(x).sum()

        backward(loss_fn, params, x)
        for p in params:
            assert isinstance(p.grad, Tensor)
