import jax
from xoe.tensor import Tensor

def backward(loss_fn, params, *args):
    param_arrays = [p._data for p in params]
    
    def pure_loss(param_arrays):
        for p, val in zip(params, param_arrays):
            p._data = val
        loss_val = loss_fn(*args)
        if isinstance(loss_val, Tensor):
            return loss_val._data
        return loss_val

    grads = jax.grad(pure_loss)(param_arrays)
    
    for p, orig, g in zip(params, param_arrays, grads):
        p._data = orig
        p.grad = Tensor(g)
