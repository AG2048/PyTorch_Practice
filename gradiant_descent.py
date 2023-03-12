import torch

# We make backpropagation algorithm to make line of ax+b manually
# y = ax + b
a = torch.rand(1)
b = torch.rand(1)

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y = torch.tensor([5.0, 7.0, 9.0, 11.0, 13.0])

num_epochs = 1000
learning_rate = 0.05


def forward(x):
    return a * x + b


# MSE Loss
def loss(y, y_pred):
    return torch.mean((y - y_pred) ** 2)


def gradient_a(x, y, y_pred):
    return torch.mean(-2 * x * (y - y_pred))


def gradient_b(x, y, y_pred):
    return torch.mean(-2 * (y - y_pred))


for epoch in range(num_epochs):
    y_pred = forward(x)
    L = loss(y, y_pred)
    da = gradient_a(x, y, y_pred)
    db = gradient_b(x, y, y_pred)
    a -= learning_rate * da
    b -= learning_rate * db
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: ")
        print(f"\ty_pred: {y_pred}")
        print(f"\tLoss: {L}")
        print(f"\ta: {a}")
        print(f"\tb: {b}")