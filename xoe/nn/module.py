from xoe.tensor import Tensor


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        params = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
            elif isinstance(v, dict):
                for item in v.values():
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def state_dict(self, prefix=""):
        state = {}
        for k, v in self.__dict__.items():
            name = f"{prefix}{k}"
            if isinstance(v, Tensor):
                state[name] = v._data
            elif isinstance(v, Module):
                state.update(v.state_dict(prefix=f"{name}."))
        return state

    def load_state_dict(self, state, prefix=""):
        for k, v in self.__dict__.items():
            name = f"{prefix}{k}"
            if isinstance(v, Tensor) and name in state:
                v._data = state[name]
            elif isinstance(v, Module):
                v.load_state_dict(state, prefix=f"{name}.")

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None
