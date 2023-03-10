import torch

# Everything in pyTorch is a tensor, a multidimensional array

# Scalar, vector, matrix, tensor

empty_tensor_0d = torch.empty(1)
print(f"empty_tensor_0d: \n{empty_tensor_0d}")

empty_tensor_1d = torch.empty(2)
print(f"empty_tensor_1d: \n{empty_tensor_1d}")

empty_tensor_2d = torch.empty(3, 4)
print(f"empty_tensor_2d: \n{empty_tensor_2d}")

empty_tensor_3d = torch.empty(3, 4, 5)
print(f"empty_tensor_3d: \n{empty_tensor_3d}")

# Create a tensor with random values

random_tensor_0d = torch.rand(1)
print(f"random_tensor_0d: \n{random_tensor_0d}")

random_tensor_1d = torch.rand(2)
print(f"random_tensor_1d: \n{random_tensor_1d}")

# Size of a tensor

print(f"Size of random_tensor_1d: {random_tensor_1d.size()}")
print(f"Size of empty_tensor_3d: {empty_tensor_3d.size()}")

# Specify the data type

random_tensor_1d = torch.rand(2, dtype=torch.float64)  # torch.float16, torch.float32, torch.float64...
print(f"random_tensor_1d: \n{random_tensor_1d}")

# Requires_grad: By default, requires_grad is False
# Any operation on a tensor with requires_grad=True will create a computational graph
x = torch.rand(2, 2, requires_grad=True)

x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(f"x: \n{x}")
print(f"y: \n{y}")

# Operations on tensors, element-wise
print(f"x + y: \n{x + y}")  # torch.add(x, y)
print(f"x - y: \n{x - y}")  # torch.sub(x, y)
print(f"x * y: \n{x * y}")  # torch.mul(x, y)
print(f"x / y: \n{x / y}")  # torch.div(x, y)

# In-place operations - original value changed.
# x.add_(y)
# x.sub_(y)
# x.mul_(y)
# x.div_(y)

# Slicing
print(f"x[:, 1]: \n{x[:, 1]}")  # Second column
print(f"x[1, :]: \n{x[1, :]}")  # Second row
print(f"x[1, 1]: \n{x[1, 1].item()}")  # Second row, second column. Convert to scalar using .item()

# Reshaping

x = torch.randn(4, 4)
print(f"x: \n{x}")
y = x.view(16)
print(f"y: \n{y}")
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(f"z: \n{z}")

# Converting a tensor to a numpy array
# <tensor_variable_name>.numpy() - torch to numpy
# torch.from_numpy() - numpy to torch
# note that the numpy array and torch tensor will share their underlying memory locations
#   (if the torch tensor is on CPU), and changing one will change the other.

# GPU:
# If you have a CUDA enabled GPU, you can move tensors to the GPU using the .to method.

if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create tensor on GPU, a tensor of ones, same size as input (x)
    x = x.to(device)                       # or just use strings ``.to("cuda")``, this moves the tensor to GPU
    z = x + y                              # z is also on GPU
    # you cannot add a tensor on GPU to a tensor on CPU, you need to move them to the same device
    # also numpy arrays cannot be used on GPU
    # so no more z.numpy()
    # but you can z.to("cpu").numpy() to move the tensor to CPU and then convert it to a numpy array



