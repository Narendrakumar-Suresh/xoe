from xoe.tensor import Tensor


def _recurse_children(module):
    for v in module.__dict__.values():
        if isinstance(v, Module):
            yield v
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, Module):
                    yield item
        elif isinstance(v, dict):
            for item in v.values():
                if isinstance(item, Module):
                    yield item


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self, recurse=True):
        seen = set()
        params = []
        for k, v in self.__dict__.items():
            if isinstance(v, Tensor) and not k.startswith("_") and id(v) not in seen:
                params.append(v)
                seen.add(id(v))
            elif recurse and isinstance(v, Module):
                for p in v.parameters(recurse=True):
                    if id(p) not in seen:
                        params.append(p)
                        seen.add(id(p))
            elif recurse and isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        for p in item.parameters(recurse=True):
                            if id(p) not in seen:
                                params.append(p)
                                seen.add(id(p))
                    elif isinstance(item, Tensor) and id(item) not in seen:
                        params.append(item)
                        seen.add(id(item))
            elif recurse and isinstance(v, dict):
                for item in v.values():
                    if isinstance(item, Module):
                        for p in item.parameters(recurse=True):
                            if id(p) not in seen:
                                params.append(p)
                                seen.add(id(p))
                    elif isinstance(item, Tensor) and id(item) not in seen:
                        params.append(item)
                        seen.add(id(item))
        return params

    def state_dict(self, prefix=""):
        state = {}
        for k, v in self.__dict__.items():
            name = f"{prefix}{k}"
            if isinstance(v, Tensor):
                state[name] = v.data.copy()
            elif isinstance(v, Module):
                state.update(v.state_dict(prefix=f"{name}."))
            elif isinstance(v, (list, tuple)):
                for idx, item in enumerate(v):
                    if isinstance(item, Tensor):
                        state[f"{name}.{idx}"] = item.data.copy()
                    elif isinstance(item, Module):
                        state.update(item.state_dict(prefix=f"{name}.{idx}."))
            elif isinstance(v, dict):
                for key, item in v.items():
                    if isinstance(item, Tensor):
                        state[f"{name}.{key}"] = item.data.copy()
                    elif isinstance(item, Module):
                        state.update(item.state_dict(prefix=f"{name}.{key}."))
        return state

    def load_state_dict(self, state, prefix=""):
        errors = []
        for k, v in self.__dict__.items():
            name = f"{prefix}{k}"
            if isinstance(v, Tensor) and name in state:
                target = state[name]
                if v.data.shape != target.shape:
                    errors.append(
                        f"Shape mismatch for '{name}': "
                        f"expected {v.data.shape}, got {target.shape}"
                    )
                    continue
                v.data = target
            elif isinstance(v, Module):
                errors.extend(v.load_state_dict(state, prefix=f"{name}."))
            elif isinstance(v, (list, tuple)):
                for idx, item in enumerate(v):
                    item_name = f"{name}.{idx}"
                    if isinstance(item, Tensor) and item_name in state:
                        target = state[item_name]
                        if item.data.shape != target.shape:
                            errors.append(
                                f"Shape mismatch for '{item_name}': "
                                f"expected {item.data.shape}, got {target.shape}"
                            )
                            continue
                        item.data = target
                    elif isinstance(item, Module):
                        errors.extend(
                            item.load_state_dict(state, prefix=f"{item_name}.")
                        )
            elif isinstance(v, dict):
                for key, item in v.items():
                    item_name = f"{name}.{key}"
                    if isinstance(item, Tensor) and item_name in state:
                        target = state[item_name]
                        if item.data.shape != target.shape:
                            errors.append(
                                f"Shape mismatch for '{item_name}': "
                                f"expected {item.data.shape}, got {target.shape}"
                            )
                            continue
                        item.data = target
                    elif isinstance(item, Module):
                        errors.extend(
                            item.load_state_dict(state, prefix=f"{item_name}.")
                        )
        expected_keys = set(self.state_dict(prefix=prefix).keys())
        provided_keys = {k for k in state if k.startswith(prefix)}
        missing = expected_keys - provided_keys
        if missing:
            for key in sorted(missing):
                errors.append(f"Missing key in state_dict: '{key}'")
        return errors

    def train(self, mode: bool = True):
        self.training = mode
        for child in _recurse_children(self):
            child.train(mode)
        return self

    def eval(self):
        return self.train(False)
