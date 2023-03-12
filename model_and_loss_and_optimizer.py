import torch
import torch.nn as nn

# For model, input and output are stored in a matrix. Each row is an example, and each column is a feature
X = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).view(5, 1)  # 5 rows, 1 column, 5 examples with 1 feature each
Y = torch.tensor([5.0, 7.0, 9.0, 11.0, 13.0]).view(5, 1)  # 5 rows, 1 column, 5 examples with 1 feature each

n_samples, n_features = X.shape  # n_samples = 5, n_features = 1

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size, bias=True)  # Linear Regression Model (y = ax + b)
# without bias, the model will be y = ax

'''
# This code is equivalent to the above code, but it is more verbose, allow more flexibility and is more readable
class Linear(nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.linear(x)
        
model = Linear(input_size, output_size)
'''


num_epochs = 1000
learning_rate = 0.05

loss = nn.MSELoss()  # Mean Squared Error Loss Function (MSE) is used

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Tell the optimizer which parameters to update

for epoch in range(num_epochs):
    # Forward Pass, predict y_pred
    y_pred = model(X)
    # Compute Loss
    L = loss(Y, y_pred)
    # Backward Pass, compute gradients
    L.backward()
    # Update weights using optimizer
    optimizer.step()
    # Zero gradients
    optimizer.zero_grad()
    if epoch % 100 == 0:
        a, b = model.parameters()
        print(f"Epoch {epoch}: ")
        print(f"\ty_pred: {y_pred}")
        print(f"\tLoss: {L.item()}")
        # print(f"\ta: {model.weight.item()}")
        # print(f"\tb: {model.bias.item()}")
        print(f"\ta: {a.item()}")
        print(f"\tb: {b.item()}")