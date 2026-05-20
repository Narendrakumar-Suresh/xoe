import jax.numpy as jnp
from xoe import Tensor
from xoe.autograd import backward

# Import the current CrossEntropyLoss (from the functional module)
from xoe.nn.functional.crossentropyloss import CrossEntropyLoss

# Let's define dummy predictions (logits/probs) and target (one-hot)
# Let's say we have 2 classes, batch size 2
pred = Tensor([[0.1, 0.9], [0.8, 0.2]])
target = Tensor([[0.0, 1.0], [1.0, 0.0]])

loss_fn_inst = CrossEntropyLoss()

# 1. Test running forward
try:
    loss = loss_fn_inst(pred, target)
    print("Forward output type:", type(loss))
    print("Forward output:", loss)
except Exception as e:
    print("Forward failed with error:", e)


# 2. Test running backward (which simulates typical training)
def loss_wrapper(pred):
    return loss_fn_inst(pred, target)


try:
    # Autograd backward uses loss_wrapper(pred)._data
    # Let's see what happens
    backward(loss_wrapper, [pred])
    print("Backward succeeded!")
except Exception as e:
    print("Backward failed with error:", type(e), e)
