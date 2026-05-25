import jax
from functools import wraps
from xoe.tensor import Tensor

def jit(fn):
    jitted_fn = None
    
    @wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal jitted_fn
        if jitted_fn is None:
            def inner(*raw_args):
                wrapped_args = [Tensor(arg) for arg in raw_args]
                res = fn(*wrapped_args, **kwargs)
                if isinstance(res, Tensor):
                    return res._data
                return res
            jitted_fn = jax.jit(inner)
            
        raw_args = [arg._data if isinstance(arg, Tensor) else arg for arg in args]
        out_data = jitted_fn(*raw_args)
        return Tensor(out_data)
        
    return wrapper
