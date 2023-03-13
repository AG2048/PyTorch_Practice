import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from sklearn.preprocessing import StandardScaler

"""
Key ideas:
    - Dataset: An abstract class representing a dataset
    - DataLoader: Iterates through a dataset and returns batches of data
    
    We will create a custom dataset class to load our data - in batches
    
    epoch is running EVERYTHING once, batch is a subset of the data, sample is a single data point
    
Small note:
    the tutorial didn't do optimization, but I did. I used the same code from logistic_regression.py
    originally the data wasn't "scaled" and it's just breaking... I don't know why. I think it's because numbers are big
    and the optimizer is having a hard time converging. I scaled the data and it works now.
    
    Step is very simple:
        1. Load data using Dataset
        2. Create DataLoader
        3. Train the model
            3.1. Create model (2 layers with sigmoid)
            3.2. Create loss and optimizer
            3.3. Train the model
"""


class WineDataSet(Dataset):
    # This is a custom dataset class that inherits from the Dataset class
    def __init__(self):
        sc = StandardScaler()
        # Initialize the data.
        # Load data to a n x m numpy array, where n is the number of samples and m-1 is the number of features
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        # Number of sample is just the number of rows
        self.n_samples = xy.shape[0]
        # Here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(sc.fit_transform(xy[:, 1:]))   # n_samples, n_features
        self.y_data = xy[:, [0]]  # n_samples, 1
        # I want to convert y_data to a list of vector of 0s and 1s. for example: 1 is [1, 0, 0]. 2 is [0, 1, 0]
        y_data = np.zeros((self.n_samples, 3))
        for i in range(self.n_samples):
            y_data[i, int(self.y_data[i]) - 1] = 1
        self.y_data = torch.from_numpy(y_data)
        # Here y_data is what we want.

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# Create dataset
dataset = WineDataSet()

# # get first sample and unpack
# # we use the __getitem__ method
# first_data = dataset[0]
# features, labels = first_data
# # print(features, labels)

# Load entire dataset with DataLoader
# dataset = where data is stored, batch_size = size of batch, shuffle = shuffle data,
#   num_workers = how many subprocesses to use for data loading
#   (not necessary for small datasets, but will speed up for large datasets)
# Shuffle: important for training, not for testing, shifts the data around
# For some reason, num_workers != 0 will crash the program
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

# # To iterate through the entire dataset, we use a for loop
# dataiter = iter(dataloader)
# # This grabs batch_size samples. in this case, 4 x vectors and 4 y vectors
# data = next(dataiter)
# features, labels = data
# print(features, labels)

# Training loop
num_epoch = 10
learning_rate = 0.01
total_samples = len(dataset)
batch_size = 4
n_iterations = math.ceil(total_samples / batch_size)

# model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear_1 = nn.Linear(input_size, 8)
        self.linear_2 = nn.Linear(8, output_size)

    def forward(self, x):
        out_1 = torch.sigmoid(self.linear_1(x))
        out_2 = torch.sigmoid(self.linear_2(out_1))
        return out_2

n_samples, n_features = dataset.x_data.shape
model = LinearRegressionModel(input_size=n_features, output_size=3)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Before Training
with torch.no_grad():
    x = dataset.x_data
    y = dataset.y_data
    y_pred = model(x)
    l = loss(y_pred, y)
    # get accuracy
    _, predicted = torch.max(y_pred.data, 1)
    _, actual = torch.max(y.data, 1)
    correct = (predicted == actual).sum().item()
    accuracy = correct / total_samples
    print(
        f'BEFORE TRAINING, loss = {l.item():.4f}, accuracy = {accuracy:.4f} ({correct}/{total_samples})')

for epoch in range(num_epoch):
    for i, (inputs, labels) in enumerate(dataloader):
        # Forward pass, backward pass, update weights
        # Forward pass
        outputs = model(inputs)
        # print(inputs)
        # print(outputs)
        # Calculate loss
        L = loss(outputs, labels)
        # Backward pass
        L.backward()
        # Update weights
        optimizer.step()
        # Zero gradients
        optimizer.zero_grad()

        # if (i+1) % 5 == 0:
        #     print(f'epoch {epoch+1}/{num_epoch}, step {i+1}/{n_iterations}, inputs {inputs.shape}')

    if (epoch) % (num_epoch//10) == 0:
        with torch.no_grad():
            x = dataset.x_data
            y = dataset.y_data
            y_pred = model(x)
            l = loss(y_pred, y)
            # get accuracy
            _, predicted = torch.max(y_pred.data, 1)
            _, actual = torch.max(y.data, 1)
            # if (epoch == 0):
            #     # print(y_pred)
            #     # print(y)
            #     print(predicted)
            #     print(actual)
            correct = (predicted == actual).sum().item()
            accuracy = correct / total_samples
            print(f'epoch {epoch+1}/{num_epoch}, loss = {l.item():.4f}, accuracy = {accuracy:.4f} ({correct}/{total_samples})')


        # torchvision.datasets.MNIST()