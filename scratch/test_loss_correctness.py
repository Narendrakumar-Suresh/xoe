import jax.numpy as jnp
from xoe import Tensor
from xoe.autograd import backward
from xoe.nn.functional.crossentropyloss import CrossEntropyLoss

# 1. Instantiate the loss
loss_fn = CrossEntropyLoss(reduction="mean")

# 2. Check logits (shape 2, 3) and targets (one-hot or index)
logits = Tensor([[1.0, 2.0, 3.0], [1.0, 5.0, 1.0]], requires_grad=True)
# Target indices
targets_index = Tensor([2, 1])
# Target probabilities
targets_prob = Tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

# 3. Compute losses
loss_index = loss_fn(logits, targets_index)
loss_prob = loss_fn(logits, targets_prob)

print("Loss with index target:", loss_index)
print("Loss with probability target:", loss_prob)

# Assert they are equal
assert jnp.allclose(loss_index._data, loss_prob._data)
print("Correctness Check: Succeeded! Both target formats yield the same loss value.")


# 4. Check that autograd backward works
# We will compute gradients with respect to logits
def loss_wrapper(logits):
    return loss_fn(logits, targets_index)


backward(loss_wrapper, [logits])
print("Gradients computed successfully!")
print("Logits gradients:\n", logits.grad)
