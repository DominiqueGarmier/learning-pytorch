from __future__ import annotations

import gzip
import math
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import requests  # type: ignore
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

DATA_PATH = Path('data')
PATH = DATA_PATH / 'mnist'

PATH.mkdir(parents=True, exist_ok=True)

URL = 'https://github.com/pytorch/tutorials/raw/master/_static/'
FILE_NAME = 'mnist.pkl.gz'

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
BATCH_SIZE = 64  # batch size
LEARNING_RATE = 0.5  # learning rate
EPOCHS = 2  # how many epochs to train for


class MyDataLoader(DataLoader):
    def __iter__(self):
        yield from super().__iter__()


def load_dataset(batch_size):
    if not (PATH / FILE_NAME).exists():
        response = requests.get(f'{URL}{FILE_NAME}')
        assert response.status_code == 200, 'could not download'

        with open(PATH / FILE_NAME, 'wb') as file:
            file.write(response.content)

    with gzip.open((PATH / FILE_NAME).as_posix(), 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    train_dl = MyDataLoader(TensorDataset(x_train, y_train))
    valid_dl = MyDataLoader(TensorDataset(x_valid, y_valid))
    return train_dl, valid_dl


class MnisitLogistic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # define all the parameters here by instantiating from nn.Paramter
        # self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        # self.bias = nn.Parameter(torch.zeros(10))

        # or by using the nn.Linear class
        self.linear = nn.Linear(784, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return x @ self.weights + self.bias
        return self.linear(x)


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1, 28, 28)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.avg_pool2d(x, 4)
        return x.view(-1, x.size(1))


def get_model(Model):
    model = Model()
    # model.to(DEVICE)
    return model, optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


def train(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(f'epoch, loss: {epoch}, {valid_loss}')


train_dataloader, valid_dataloader = load_dataset(batch_size=BATCH_SIZE)
model, opt = get_model(MnisitLogistic)
loss_func = nn.functional.cross_entropy
train(EPOCHS, model, loss_func, opt, train_dataloader, valid_dataloader)
