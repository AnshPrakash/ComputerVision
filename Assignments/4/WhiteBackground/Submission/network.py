'''
 Code of network.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import f1_score


class Net(nn.Module):
  def __init__(self, num_classes=4):
    super(Net, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 5, stride=2))
    self.layer2 = nn.Sequential(
        nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=5, stride=3))
    # self.layer3 = nn.Sequential(
    #     nn.Conv2d(6, 9, kernel_size=3, stride=1, padding=0),
    #     nn.BatchNorm2d(9),
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=2, stride=1))
    self.fc1 = nn.Linear(400,128)
    self.fc2 = nn.Linear(128,32)
    self.fc3 = nn.Linear(32,4)

  def forward(self, x):
    # print("**")
    # print(x.size())
    out = self.layer1(x)
    # print(out.size())
    # out = F.dropout(out,p=0.1)
    out = self.layer2(out)
    # out = F.dropout(out,p=0.1)
    # print(out.size())

    # out = self.layer3(out)
    # print(out.size())
    out = out.reshape(out.size(0), -1)
    out = self.fc1(out)
    # print(out.size())
    out = self.fc2(out)
    out = self.fc3(out)
    # print("=====")
    return out
