import jax
from xoe.tensor import Tensor


def backward(loss_fn, params, *args):
    param_arrays = [p.data for p in params]

    def pure_loss(flat_params):
        for p, val in zip(params, flat_params):
            p.data = val
        result = loss_fn(*args)
        return result.data if isinstance(result, Tensor) else result

    grads = jax.grad(pure_loss)(param_arrays)

    for p, orig, g in zip(params, param_arrays, grads):
        p.data = orig
        p.grad = Tensor._wrap(g)
