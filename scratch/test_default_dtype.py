import xoe

# Check current default
print(f"Initial default dtype: {xoe.get_default_dtype()}")

# Create tensor (should be f32)
x = xoe.Tensor([1, 2, 3])
print(f"x dtype: {x.dtype}")

# Set new default
xoe.set_default_dtype(xoe.f16)
print(f"New default dtype: {xoe.get_default_dtype()}")

# Create tensor (should be f16)
y = xoe.Tensor([1, 2, 3])
print(f"y dtype: {y.dtype}")

# Check random (should be f16)
from xoe import random

z = xoe.Tensor(random.randn((2, 2)))
print(f"z (random) dtype: {z.dtype}")
