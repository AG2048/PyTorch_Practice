import torch
import torch.nn as nn

X = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
Y = torch.tensor([5.0, 7.0, 9.0, 11.0, 13.0])

a = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)

num_epochs = 1000
learning_rate = 0.05


def forward(x):
    return a * x + b


loss = nn.MSELoss()  # Mean Squared Error Loss Function (MSE) is used

optimizer = torch.optim.SGD([a, b], lr=learning_rate)  # Stochastic Gradient Descent (SGD) is used

for epoch in range(num_epochs):
    # Forward Pass, predict y_pred
    y_pred = forward(X)
    # Compute Loss
    L = loss(Y, y_pred)
    # Backward Pass, compute gradients
    L.backward()
    # Update weights using optimizer
    optimizer.step()
    # Zero gradients
    optimizer.zero_grad()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: ")
        print(f"\ty_pred: {y_pred}")
        print(f"\tLoss: {L.item()}")
        print(f"\ta: {a}")
        print(f"\tb: {b}")