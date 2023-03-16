import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from PIL import Image
import pygame


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

im = Image.open("image/pixil-frame-0.png")
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






















dset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())


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


dataset = ImageDataSet(dset)

# After Training
with torch.no_grad():
    x = dataset.x_data
    y = dataset.y_data
    y_pred = model(x)
    l = None
    # get accuracy
    _, predicted = torch.max(y_pred.data, 1)
    _, actual = torch.max(y.data, 1)
    correct = (predicted == actual).sum().item()
    accuracy = correct / len(dataset)
    print(
        f'AFTER TRAINING, loss = {l}, accuracy = {accuracy:.4f} ({correct}/{len(dataset)})')



# Make a pygame loop that makes a 280x280 canvas with 10x10 pixels. whenever user click left mouse button, it will make a 10x10 pixel black. when user click right button, it clears the canvas. The model outputs the predicted number on the top right of the screen.


title = "number_pred"  # the title of the window
cell_size = 10  # define the size of a block
bg_size = (28*cell_size, 28*cell_size)  # the size of the background
WHITE = (255, 255, 255)  # tell comp what is white
BLACK = (0, 0, 0)
COL = (255, 255, 0)


# set up window
pygame.init()  # mush have, initialize the game
pygame.display.set_caption(title)  # Set the display, with the caption title (Andy_Snake)
caption = pygame.display.set_mode(bg_size)  # set the display with the size of bg_size


def draw_rect(colour, coordinate):
    pygame.draw.rect(caption, colour, pygame.Rect(coordinate[0], coordinate[1], cell_size, cell_size))

def end_game():
    pygame.quit()


def main():
    board = [[0 for i in range(28)] for j in range(28)]
    mouse_down = False
    myFont = pygame.font.SysFont("Times New Roman", 18)
    pred_number = 0
    last_pos = (-1, -1)
    while True:
        caption.blit(myFont.render(str(pred_number), True, (255, 255, 255)), (0, 0))
        pygame.display.update()
        caption.fill(BLACK)  # it covers the previous white blocks
        clock = pygame.time.Clock()
        clock.tick(9999999)  # set the fps to 10

        if not mouse_down:
            output = model(torch.tensor(board, dtype=torch.float32).view(1, 1, 28, 28))
            pred_number = torch.max(output.data, 1)[1].item()
            # print(pred_number)
            # print("blit")


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_down = True
                if event.button == 3:
                    # for row in board:
                        # print(row)
                    board = [[0 for i in range(28)] for j in range(28)]
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down = False
        if mouse_down:
            x, y = pygame.mouse.get_pos()
            try:
                if (y//cell_size, x//cell_size) != last_pos:
                    # board[y//cell_size][x//cell_size] = 1
                    # board[y // cell_size+1][x // cell_size] = min(1, board[y // cell_size+1][x // cell_size] + 0.3)
                    # board[y // cell_size][x // cell_size+1] = min(1, board[y // cell_size][x // cell_size+1] + 0.3)
                    # board[y // cell_size - 1][x // cell_size] = min(1, board[y // cell_size - 1][x // cell_size] + 0.3)
                    # board[y // cell_size][x // cell_size-1] = min(1, board[y // cell_size][x // cell_size-1] + 0.3)
                    # board[y // cell_size + 1][x // cell_size + 1] = min(1, board[y // cell_size + 1][x // cell_size + 1] + 0.2)
                    # board[y // cell_size - 1][x // cell_size - 1] = min(1, board[y // cell_size - 1][x // cell_size - 1] + 0.2)
                    # board[y // cell_size + 1][x // cell_size - 1] = min(1, board[y // cell_size + 1][x // cell_size - 1] + 0.2)
                    # board[y // cell_size - 1][x // cell_size + 1] = min(1, board[y // cell_size - 1][x // cell_size + 1] + 0.2)
                    board[y // cell_size][x // cell_size] = 1
                    board[y // cell_size + 1][x // cell_size] = 1
                    board[y // cell_size][x // cell_size + 1] = 1
                    board[y // cell_size + 1][x // cell_size + 1] = 1
                    if x // cell_size - 1 >= 0:
                        board[y // cell_size][x // cell_size - 1] = 1
                        board[y // cell_size + 1][x // cell_size - 1] = 1
                    if y // cell_size - 1 >= 0:
                        board[y // cell_size - 1][x // cell_size] = 1
                        board[y // cell_size - 1][x // cell_size + 1] = 1
                    if x // cell_size - 1 >= 0 and y // cell_size - 1 >= 0:
                        board[y // cell_size - 1][x // cell_size - 1] = 1
                    last_pos = (y//cell_size, x//cell_size)

            except:
                pass

        for i in range(28):
            for j in range(28):
                if board[j][i] != 0:
                    draw_rect(list(torch.tensor(WHITE) * board[j][i]), (i*cell_size, j*cell_size))


if __name__ == '__main__':
    main()