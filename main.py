import jax.numpy as jnp
import xoe
from xoe import Tensor
from xoe.nn.module import Module
from xoe.nn.linear import Linear
from xoe.nn.sequential import Sequential
from xoe.nn.layernorm import LayerNorm
from xoe.nn.dropout import Dropout
from xoe.nn.activations import ReLU
import xoe.nn.init as init
from xoe.autograd import backward
from xoe.optim.adam import Adam

class MLP(Module):
    def __init__(self):
        self.net = Sequential(
            Linear(2, 4),
            LayerNorm(4),
            ReLU(),
            Dropout(0.1),
            Linear(4, 1)
        )

    def forward(self, x):
        return self.net(x)

x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = Tensor([[0.0], [1.0], [1.0], [0.0]])

model = MLP()

init.kaiming_uniform_(model.net[0].W, nonlinearity='relu')
init.zeros_(model.net[0].b)
init.xavier_uniform_(model.net[4].W)
init.zeros_(model.net[4].b)

@xoe.jit
def compute_loss(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred - target
    return (diff * diff).mean()

def train_loss_fn(x):
    model.net[3].training = True
    pred = model(x)
    return compute_loss(pred, y)

def eval_loss_fn(x):
    model.net[3].training = False
    pred = model(x)
    return compute_loss(pred, y)

optimizer = Adam(model.parameters(), lr=0.01)

for i in range(10):
    optimizer.zero_grad()
    backward(train_loss_fn, model.parameters(), x)
    optimizer.step()
    print(f"step {i + 1} loss: {eval_loss_fn(x)}")
