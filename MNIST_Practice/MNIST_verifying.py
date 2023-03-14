import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from PIL import Image


class CNNModel(nn.Module):
    def __init__(self, image_height, image_width, n_classes):
        super(CNNModel, self).__init__()
        self.conv_1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        output_width = (image_width - 3 + 2 * 1) / 1 + 1
        output_height = (image_height - 3 + 2 * 1) / 1 + 1
        num_channels = 16
        # output width = (W - F + 2P) / S + 1 = (28 - 3 + 2 * 1) / 1 + 1 = 28
        # next layer: 28 / 2 = 14 because of maxpool with kernel_size=2, stride=2
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        output_width = output_width / 2
        output_height = output_height / 2
        num_channels = 16
        # output width = (14 - 3 + 2 * 1) / 1 + 1 = 14
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        output_width = (output_width - 3 + 2 * 1) / 1 + 1
        output_height = (output_height - 3 + 2 * 1) / 1 + 1
        num_channels = 32
        # next layer: 14 / 2 = 7 because of maxpool with kernel_size=2, stride=2
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
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
        out_5 = out_4.reshape(out_4.size(0), -1)
        out_6 = self.fc_1(out_5)
        return out_6


model = torch.load('model/MNIST_model.pt')
model.eval()

im = Image.open(r"image/pixil-frame-0.png")
px = im.load()
pixel_map = []
for i in range(28):
    row = []
    for j in range(28):
        row.append(px[j, i][3]/255)
    pixel_map.append(row)

pixel_map = torch.tensor(pixel_map, dtype=torch.float32)

pred = model(pixel_map.view(1, 1, 28, 28))
# convert to a regular python list
print(pred.data.numpy().tolist()[0])
print(torch.max(pred.data, 1)[1].item())
