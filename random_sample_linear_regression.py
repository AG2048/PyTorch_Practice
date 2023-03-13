import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Grab the data from sklearn datasets
# This generates two 100x1 numpy arrays
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20)

# Convert numpy arrays to torch tensors
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32)).view(-1, 1)

n_samples, n_features = X.shape

input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size, bias=True)

learning_rate = 0.01
num_epochs = 1000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    # Forward Pass, predict y_pred
    y_pred = model(X)
    # Compute Loss
    L = loss(y, y_pred)
    # Backward Pass, compute gradients
    L.backward()
    # Update weights using optimizer
    optimizer.step()
    # Zero gradients
    optimizer.zero_grad()
    if epoch % 100 == 0:
        a, b = model.parameters()
        print(f"Epoch {epoch}: ")
        # print(f"\ty_pred: {y_pred}")
        print(f"\tLoss: {L.item()}")
        # print(f"\ta: {a.item()}")
        # print(f"\tb: {b.item()}")

print(f"Final Loss: {L.item()}")
print(f"Final a: {a.item()}")
print(f"Final b: {b.item()}")
# Plot the data and the model
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show(grid=True)
