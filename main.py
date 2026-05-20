from xoe import Tensor
from xoe.nn.linear import Linear
from xoe.nn.module import Module
from xoe.nn.activations import relu
from xoe.autograd import backward
from xoe.optim.adam import Adam


# simple 2-layer MLP
class MLP(Module):
    def __init__(self):
        self.l1 = Linear(2, 4)
        self.l2 = Linear(4, 1)

    def forward(self, x):
        x = relu(self.l1(x))
        return self.l2(x)


# dummy data — XOR-like
x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = Tensor([[0.0], [1.0], [1.0], [0.0]])

model = MLP()


# loss fn
def loss_fn(x):
    pred = model(x)
    diff = pred - y
    return (diff * diff).mean()


# training loop
optimizer = Adam(model.parameters(), lr=0.01)

for i in range(10):
    optimizer.zero_grad()
    backward(loss_fn, model.parameters(), x)
    optimizer.step()
    print(f"step {i + 1} loss: {loss_fn(x)}")
