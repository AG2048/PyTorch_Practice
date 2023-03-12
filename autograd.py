import torch

# This tells pyTorch to build a graph, and keep track of operations on tensors
x = torch.rand(3, requires_grad=True)
y = x + 2
print(f"x: \n{x}")
print(f"y: \n{y}")
print(f"y.grad_fn: \n{y.grad_fn}")

z = y * y * 2
print(f"z: \n{z}")
print(f"z.grad_fn: \n{z.grad_fn}")
output = z.mean()
print(f"output: \n{output}")
print(f"output.grad_fn: \n{output.grad_fn}")

# To get the gradients, we need to call .backward() on the **output** tensor
output.backward()
# After calling .backward(), the gradients are stored in the .grad attribute of the respective tensor
# In this case, the operation is: (2(a+2)^2 + 2(b+2)^2 + 2(c+2)^2) / 3. So the gradients are:
# 4(a+2) / 3, 4(b+2) / 3, 4(c+2) / 3
print(f"x.grad: \n{x.grad}")
print(f"expected gradients: \n{(4 * (x + 2) / 3).detach()}")

# Also, if there are multiple outputs, we can pass a vector to .backward() to specify a gradient for each output
# the tensor scales the gradients before they are accumulated
x = torch.rand(3, requires_grad=True)
y = x * 2
y.backward(torch.tensor([1.0, 0.1, 0.01]))  # this is the gradient for each output
print(f"x.grad: \n{x.grad}")
print(f"we expect: \n{2 * torch.ones(3)}")

# To stop pyTorch from tracking history on Tensors with .requires_grad=True, we can wrap the code block in
# with torch.no_grad():
#     ...
# This will prevent the computation from being tracked, and will save memory
x = torch.rand(3, requires_grad=True)
x.requires_grad_(False)  # this is the same as x.requires_grad = False
x.requires_grad = True
with torch.no_grad():
    y = x * 2  # y will not be tracked because it is in a with torch.no_grad() block
print(f"y.grad_fn: \n{y.grad_fn}")

y = x.detach()  # y will not be tracked because it is detached from the computation graph
print(f"y.grad_fn: \n{y.grad_fn}")

# Note that backward() accumulates the gradients into the .grad attribute, so if we want to compute the gradients
# for a new output, we need to zero the existing gradients first
x = torch.rand(3, requires_grad=True)
y = x * 2
y.backward(torch.ones(3))
print(f"x.grad: \n{x.grad}")
y = x * 2
y.backward(torch.ones(3))
print(f"x.grad: \n{x.grad}")  # this is not what we want, we want the gradients to be 2, 2, 2
x.grad.zero_()  # this zeros the gradients
y = x * 2
y.backward(torch.ones(3))
print(f"x.grad: \n{x.grad}")  # now we get the correct gradients