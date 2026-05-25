import xoe
from xoe import Tensor, backward
import xoe.nn as nn
from xoe.nn.functional import MSELoss
from xoe.optim import AdamW

# model
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.GELU(),
    nn.LayerNorm(16),
    nn.Linear(16, 1),
)

# data
x = Tensor([[1.0, 2.0, 3.0, 4.0]] * 8)
y = Tensor([[1.0]] * 8)

# loss + optimizer
criterion = MSELoss()
optimizer = AdamW(model.parameters(), lr=1e-3)

# training loop
for step in range(20):
    optimizer.zero_grad()
    
    def loss_fn(x):
        return criterion(model(x), y)
    
    backward(loss_fn, model.parameters(), x)
    optimizer.step()
    
    print(f"step {step:02d} | loss: {loss_fn(x)}")