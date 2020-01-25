"""Defines the DesiCNN"""

import torch
import torch.nn as nn


class DesiCNN(nn.Module):
    """INPUT - 2DCONV - 2DCONV - FCN - FCN"""

    def __init__(self):
        """
        Builds the network structure with the provided parameters
        """

        super().__init__()

        # 3D Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=8, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=7, kernel_size=5, padding=0)

        # Maxpool
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Fully Convolutional layers
        self.fcn1 = nn.Linear(in_features=1008, out_features=256)
        self.fcn2 = nn.Linear(in_features=256, out_features=16)

        # Non-linearities
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        # 1st layer
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # 2nd layer
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # 3rd layer
        x = x.view(x.shape[0], -1)
        x = self.fcn1(x)

        # 4th layer
        x = self.fcn2(x)

        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)