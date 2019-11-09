import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import f1_score


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 8, kernel_size = 3,stride =2,padding =(0,0))
    self.bn2d_1 = nn.BatchNorm2d(8)
    self.pool1 = nn.MaxPool2d(kernel_size=2,stride = 2)
    self.conv2 = nn.Conv2d(in_channels = 8,out_channels = 16, kernel_size = 3,stride =2,padding =(0,0))
    self.bn2d_2 = nn.BatchNorm2d(16)
    self.pool2 = nn.MaxPool2d(kernel_size=2,stride = 2)
    self.fc = nn.Linear(64,50)
    # self.bn1 = nn.BatchNorm1d(60)
    self.out_layer = nn.Linear(50, 4)
    
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.bn2d_1(x)
    x = self.pool1(x)
    x = F.relu(self.conv2(x))
    x = self.bn2d_2(x)
    x = self.pool2(x)
    x = x.view(-1, self.num_flat_features(x))
    # print(x)
    x = F.relu(self.fc(x))
    # x = self.bn1(x)
    # print("After Hideen Layer",x)
    x = self.out_layer(x)
    # print("After Output layer",x)
    # x = F.log_softmax(x, dim=1)
    # x = F.softmax(x,dim=1) # for cross entropyloss don't do softmax as it internally does it
    # print(x)
    return x

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:       # Get the products
      num_features *= s
    return num_features
