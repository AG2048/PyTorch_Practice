# Common Steps:
# 1. Load the data
# 2. Create the model (input, output, forward pass
# 3. Train the model (loss, optimizer, training loop - forward, backward, update)

# Major difference between logistic regression and linear regression:
#   logistic regression is used for classification
#       - sigmoid function is used for classification
#       - cross entropy loss is used as loss function
#   linear regression is used for regression
#       - linear function is used for regression
#       - MSE loss is used as loss function

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler  # Scale the data for us
from sklearn.model_selection import train_test_split  # Split the data into training and testing (prevent overfitting)

'''Load Data'''
bc = datasets.load_breast_cancer()  # minor problem, we can predict cancer from input
X, y = bc.data, bc.target
n_samples, n_features = X.shape  # Again, we have 569 samples and 30 features (sample x feature matrix)

# Split the data into training and testing (prevent overfitting)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)  # 80% training, 20% testing

# Scale the data for us (mean = 0, std = 1)
# This is important because we want to make sure that the data is centered around 0
# Usually done for logistic regression and neural networks
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Fit the scaler to the training data and transform it
X_test = sc.transform(X_test)  # Transform the testing data

# Convert numpy arrays to torch tensors, we need to convert to float32 or else np is in double.
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# Reshape the data
y_train = y_train.view(y_train.shape[0], 1) # 1 column, n rows (output is only 1 value) (column vectors (nx1 matrix))
y_test = y_test.view(y_test.shape[0], 1)


'''Model'''
# f = wx + b, sigmoid at the end


class Model(nn.Module):
    # Create the model (input, output, forward pass)
    def __init__(self, n_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_features, 1)  # input, output

    def forward(self, x):
        # y = sigmoid(wx + b)
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = Model(n_features)  # 30 input, 1 output


'''loss and optimizer'''
learning_rate = 0.01
loss = nn.BCELoss()  # Binary Cross Entropy Loss. This is the loss function for logistic regression
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Stochastic Gradient Descent

'''training loop'''
# I want to see how the loss changes over time
epoch_list = []
loss_list = []
acc_list = []

num_epochs = 1000
for epoch in range(num_epochs):
    # Note it would have been better to alternate the training data set, or it's possible to get stuck in a local minima
    # forward pass
    y_pred = model(X_train)

    # loss
    L = loss(y_pred, y_train)

    # backward pass
    L.backward()

    # update
    optimizer.step()
    optimizer.zero_grad()

    # store the loss for plotting
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_class = y_pred.round()  # Round the predictions to the closest integer
        acc = y_pred_class.eq(y_test).sum() / float(
            y_test.shape[0])  # sum of correct predictions / number of predictions
    epoch_list.append(epoch)
    loss_list.append(L.item())
    acc_list.append(acc)

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch}: w = {w[0][0].item():.3f}, loss = {L.item():.8f}")
        print(f"accuracy = {acc:.4f}")

with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = y_pred.round()  # Round the predictions to the closest integer
    acc = y_pred_class.eq(y_test).sum() / float(y_test.shape[0])  # sum of correct predictions / number of predictions
    print(f"accuracy = {acc:.4f}")

# Plot the loss
plt.plot(epoch_list, loss_list, 'r.')
plt.plot(epoch_list, acc_list, 'b.')
plt.xlabel("epoch")
plt.legend(["loss", "accuracy"])
plt.show()
