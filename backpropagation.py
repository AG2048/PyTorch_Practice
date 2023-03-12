import torch

# Trying to make a backpropagation algorithm to make line of ax+b
# y = ax + b

a = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y = torch.tensor([5.0, 7.0, 9.0, 11.0, 13.0])

# y_pred = a * x + b
# loss = torch.sum((y - y_pred) ** 2)
# print(f"Loss: {loss.item()}")
# loss.backward()
# print(f"Gradient of a: {a.grad}")
# print(f"Gradient of b: {b.grad}")
# # Update weights, must use torch.no_grad() to prevent tracking gradients
# with torch.no_grad():
#     a -= 0.01 * a.grad
#     b -= 0.01 * b.grad
# a.grad.zero_()
# b.grad.zero_()
# print(f"a: {a}")
# print(f"b: {b}")
# y_pred = a * x + b
# loss = torch.sum((y - y_pred) ** 2)
# print(f"Loss: {loss.item()}")

num_epochs = 1000
learning_rate = 0.01
for epoch in range(num_epochs):
    y_pred = a * x + b
    loss = torch.sum((y - y_pred) ** 2)
    loss.backward()
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
    a.grad.zero_()
    b.grad.zero_()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: ")
        print(f"\ty_pred: {y_pred}")
        print(f"\tLoss: {loss.item()}")
        print(f"\ta: {a}")
        print(f"\tb: {b}")

y_pred = a * x + b
print(f"End Result: ")
print(f"\ty_pred: {y_pred}")
print(f"\tLoss: {loss.item()}")
print(f"\ta: {a}")
print(f"\tb: {b}")

