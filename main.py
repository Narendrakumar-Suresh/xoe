from xoe.nn.layernorm import LayerNorm
import xoe
from xoe import Tensor, backward
import xoe.nn as nn
from xoe.nn.functional import MSELoss
from xoe.optim import AdamW

class FNN(nn.Module):
    def __init__(self,indim=4,outdim=16):
        self.l1=nn.Linear(indim,outdim)
        self.l2=nn.Linear(outdim,indim)
        self.gelu=nn.GELU()
        self.norm=nn.LayerNorm(outdim)
    
    def forward(self,x):
        x=self.gelu(self.l1(x))
        x=self.l2(self.norm(x))
        return x

model=FNN()

# model = nn.Sequential(
#     nn.Linear(4, 16),
#     nn.GELU(),
#     nn.LayerNorm(16),
#     nn.Linear(16, 1),
# )

x = Tensor([[1.0, 2.0, 3.0, 4.0]] * 8)
y = Tensor([[1.0]] * 8)

criterion = MSELoss()
params = model.parameters()
optimizer = AdamW(params, lr=1e-3)


def loss_fn(x):
    return criterion(model(x), y)


for step in range(20):
    loss, grads = backward(loss_fn, params, x)
    new_data = optimizer.step([p.data for p in params], grads)
    for p, nd in zip(params, new_data):
        p.data = nd
    print(f"step {step:02d} | loss: {loss}")
