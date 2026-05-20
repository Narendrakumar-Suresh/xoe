from xoe.tensor import Tensor, set_default_dtype, get_default_dtype
import jax.numpy as jnp

# Dtypes - aliased from JAX with zero overhead
float32 = jnp.float32
float16 = jnp.float16
bfloat16 = jnp.bfloat16
int32 = jnp.int32
int16 = jnp.int16

# Shortcuts
f32 = float32
f16 = float16
bf16 = bfloat16
i32 = int32
i16 = int16
