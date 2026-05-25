from xoe.tensor import Tensor


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        params = []
        for k, v in self.__dict__.items():
            if isinstance(v, Tensor):
                if v.requires_grad:
                    params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
            elif isinstance(v, dict):
                for item in v.values():
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
        return params

    def state_dict(self, prefix=""):
        state = {}
        for k, v in self.__dict__.items():
            name = f"{prefix}{k}"
            if isinstance(v, Tensor):
                state[name] = v._data
            elif isinstance(v, Module):
                state.update(v.state_dict(prefix=f"{name}."))
            elif isinstance(v, (list, tuple)):
                for idx, item in enumerate(v):
                    if isinstance(item, Tensor):
                        state[f"{name}.{idx}"] = item._data
                    elif isinstance(item, Module):
                        state.update(item.state_dict(prefix=f"{name}.{idx}."))
            elif isinstance(v, dict):
                for key, item in v.items():
                    if isinstance(item, Tensor):
                        state[f"{name}.{key}"] = item._data
                    elif isinstance(item, Module):
                        state.update(item.state_dict(prefix=f"{name}.{key}."))
        return state

    def load_state_dict(self, state, prefix=""):
        for k, v in self.__dict__.items():
            name = f"{prefix}{k}"
            if isinstance(v, Tensor) and name in state:
                v._data = state[name]
            elif isinstance(v, Module):
                v.load_state_dict(state, prefix=f"{name}.")
            elif isinstance(v, (list, tuple)):
                for idx, item in enumerate(v):
                    item_name = f"{name}.{idx}"
                    if isinstance(item, Tensor) and item_name in state:
                        item._data = state[item_name]
                    elif isinstance(item, Module):
                        item.load_state_dict(state, prefix=f"{item_name}.")
            elif isinstance(v, dict):
                for key, item in v.items():
                    item_name = f"{name}.{key}"
                    if isinstance(item, Tensor) and item_name in state:
                        item._data = state[item_name]
                    elif isinstance(item, Module):
                        item.load_state_dict(state, prefix=f"{item_name}.")

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None
