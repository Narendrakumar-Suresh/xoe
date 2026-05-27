import jax
from xoe.tensor import Tensor


def backward(loss_fn, params, *args, **kwargs):
    saved = [(p, p.data) for p in params]
    param_arrays = [p.data for p in params]

    def pure_loss(flat_params):
        for p, val in zip(params, flat_params):
            p.data = val
        result = loss_fn(*args, **kwargs)
        if isinstance(result, Tensor):
            return result.data
        return result

    try:
        loss_val, grads = jax.value_and_grad(pure_loss)(param_arrays)
    finally:
        for p, orig in saved:
            p.data = orig

    return Tensor._wrap(loss_val), grads
