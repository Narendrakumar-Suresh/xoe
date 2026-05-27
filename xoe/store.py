from safetensors.numpy import save_file, load_file
import jax
import jax.numpy as jnp
from xoe.nn.module import Module


def save_as_safetensor(module: Module, path: str):
    state = module.state_dict()
    np_state = {k: jax.device_get(v) for k, v in state.items()}
    save_file(np_state, path)


def load_from_safetensor(module: Module, path: str):
    np_state = load_file(path)
    jax_state = {k: jnp.array(v) for k, v in np_state.items()}
    errors = module.load_state_dict(jax_state)
    if errors:
        for err in errors:
            print(f"Warning: {err}")
