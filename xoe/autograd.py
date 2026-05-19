import jax
from xoe.tensor import Tensor

def backward(loss_fn, params, *args):
    param_arrays = [p._data for p in params]

    def pure_loss(param_arrays):
        for p, arr in zip(params, param_arrays):
            p._data = arr
        return loss_fn(*args)._data

    grads = jax.grad(pure_loss)(param_arrays)
    for p, orig, g in zip(params, param_arrays, grads):
        p._data = orig  # ← restore
        p.grad = Tensor(g)