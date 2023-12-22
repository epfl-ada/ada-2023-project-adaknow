import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import argparse

class ActorNet1(nn.Module):
    def __init__(self):
        super(ActorNet1, self).__init__()
        self.fc1 = nn.Linear(38, 128)  # Adjust in_features to match your number of input features
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid activation for binary classification
        return x



class ActorNet2(nn.Module):
    def __init__(self):
        super(ActorNet2, self).__init__()
        self.fc1 = nn.Linear(38, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        x = torch.sigmoid(self.fc3(x))  # Sigmoid for binary classification
        return x

