import jax
from functools import wraps
from xoe.tensor import Tensor


def _is_tensor(x):
    return isinstance(x, Tensor)


def _to_raw(x):
    return x.data if _is_tensor(x) else x


def _from_raw(x):
    return Tensor._wrap(x)


def jit(fn):
    jitted_fn = None
    compiled_kwargs = None

    @wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal jitted_fn, compiled_kwargs

        if jitted_fn is None or kwargs != compiled_kwargs:

            def inner(*raw_args):
                wrapped_args = tuple(Tensor._wrap(arg) for arg in raw_args)
                res = fn(*wrapped_args, **kwargs)
                if isinstance(res, Tensor):
                    return res.data
                if isinstance(res, (tuple, list)):
                    return type(res)(r.data if _is_tensor(r) else r for r in res)
                return res

            jitted_fn = jax.jit(inner)
            compiled_kwargs = dict(kwargs)

        raw_args = [_to_raw(arg) for arg in args]
        out = jitted_fn(*raw_args)

        if isinstance(out, (tuple, list)):
            return type(out)(_from_raw(o) for o in out)
        return _from_raw(out)

    return wrapper
