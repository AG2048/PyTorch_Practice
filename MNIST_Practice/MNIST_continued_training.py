import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

dset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())


class ImageDataSet(Dataset):
    def __init__(self, dset):
        # x = []
        # y = []
        # for data in dset:
        #     x.append(data[0][0])
        #     y.append(data[1])
        # self.x_data = torch.stack(x)
        # self.y_data = torch.tensor(y)
        # self.n_samples = len(self.x_data)
        self.x_data, self.y_data = dset.data, dset.targets
        self.n_samples = len(self.x_data)
        self.x_data = torch.tensor(self.x_data, dtype=torch.float32)
        self.x_data = self.x_data / 255
        self.x_data = self.x_data.reshape(self.n_samples, 1, 28, 28)
        y_data = np.zeros((self.n_samples, 10))
        for i in range(self.n_samples):
            y_data[i, int(self.y_data[i])] = 1
        self.y_data = torch.from_numpy(y_data)

        # # only take 1000 samples to train for now:
        # self.x_data = self.x_data[:1000]
        # self.y_data = self.y_data[:1000]
        # self.n_samples = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# Training Setup
num_epoch = 1
batch_size = 10
learning_rate = 0.01
dataset = ImageDataSet(dset)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / batch_size)


class CNNModel(nn.Module):
    def __init__(self, image_height, image_width, n_classes):
        super(CNNModel, self).__init__()
        self.conv_1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=1)
        output_width = (image_width - 5 + 2 * 1) / 1 + 1
        output_height = (image_height - 5 + 2 * 1) / 1 + 1
        num_channels = 16
        # output width = (W - F + 2P) / S + 1 = (28 - 3 + 2 * 1) / 1 + 1 = 28
        # next layer: 28 / 2 = 14 because of maxpool with kernel_size=2, stride=2
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        output_width = output_width / 2
        output_height = output_height / 2
        num_channels = 16
        # output width = (14 - 3 + 2 * 1) / 1 + 1 = 14
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=1)
        output_width = (output_width - 4 + 2 * 1) / 1 + 1
        output_height = (output_height - 4 + 2 * 1) / 1 + 1
        num_channels = 32
        # next layer: 14 / 2 = 7 because of maxpool with kernel_size=2, stride=2
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        output_width = output_width / 2
        output_height = output_height / 2
        self.conv_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        output_width = (output_width - 3 + 2 * 1) / 1 + 1
        output_height = (output_height - 3 + 2 * 1) / 1 + 1
        num_channels = 32
        # next layer: 14 / 2 = 7 because of maxpool with kernel_size=2, stride=2
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        output_width = output_width / 2
        output_height = output_height / 2
        num_channels = 32
        # input to fc_1 is 7 * 7 * 32 because: we have 7x7 image and 32 channels
        self.fc_1 = nn.Linear(int(output_width * output_width * num_channels), n_classes)

    def forward(self, x):
        out_1 = torch.relu(self.conv_1(x))
        out_2 = self.pool_1(out_1)
        out_3 = torch.relu(self.conv_2(out_2))
        out_4 = self.pool_2(out_3)
        out_5 = torch.relu(self.conv_3(out_4))
        out_6 = self.pool_3(out_5)
        out_7 = out_6.reshape(out_6.size(0), -1)
        out_8 = self.fc_1(out_7)
        return out_8


model = torch.load('model/MNIST_model.pt')
model.eval()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
# Before Training
# with torch.no_grad():
#     x = dataset.x_data
#     # print(x[0])
#     y = dataset.y_data
#     y_pred = model(x)
#     # print(y[0])
#     # print(y_pred[0])
#     l = criterion(y_pred, y)
#     # get accuracy
#     _, predicted = torch.max(y_pred.data, 1)
#     _, actual = torch.max(y.data, 1)
#     correct = (predicted == actual).sum().item()
#     accuracy = correct / total_samples
#     print(
#         f'BEFORE TRAINING, loss = {l.item():.4f}, accuracy = {accuracy:.4f} ({correct}/{total_samples})')

for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(dataloader):
        outputs = model(images)
        L = criterion(outputs, labels)
        L.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f'epoch {epoch + 1}/{num_epoch}, step {i + 1}/{n_iterations}, loss = {L.item():.4f}')

    # if (epoch+1) % (num_epoch // 10) == 0:
    #     with torch.no_grad():
    #         x = dataset.x_data
    #         y = dataset.y_data
    #         y_pred = model(x)
    #         l = criterion(y_pred, y)
    #         # get accuracy
    #         _, predicted = torch.max(y_pred.data, 1)
    #         _, actual = torch.max(y.data, 1)
    #         # if (epoch == 0):
    #         #     # print(y_pred)
    #         #     # print(y)
    #         #     print(predicted)
    #         #     print(actual)
    #         correct = (predicted == actual).sum().item()
    #         accuracy = correct / total_samples
    #         print(
    #             f'epoch {epoch + 1}/{num_epoch}, loss = {l.item():.4f}, accuracy = {accuracy:.4f} ({correct}/{total_samples})')

# After Training
with torch.no_grad():
    x = dataset.x_data
    y = dataset.y_data
    y_pred = model(x)
    l = criterion(y_pred, y)
    # get accuracy
    _, predicted = torch.max(y_pred.data, 1)
    _, actual = torch.max(y.data, 1)
    correct = (predicted == actual).sum().item()
    accuracy = correct / total_samples
    print(
        f'AFTER TRAINING, loss = {l.item():.4f}, accuracy = {accuracy:.4f} ({correct}/{total_samples})')

# Save Model
torch.save(model, 'model/MNIST_model.pt')
