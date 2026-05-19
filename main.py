from xoe.tensor import Tensor

a = Tensor([[1.0, 2.0], [3.0, 4.0]])
b = Tensor([[5.0, 6.0], [7.0, 8.0]])

print(a.T)
print(a @ b)
print(a + b)
print(a.shape)
