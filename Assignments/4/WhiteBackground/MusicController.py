from imutils.video import count_frames
import numpy as np
import cv2
import sys
import os

'''
  0 : BackGround
  1 : Next
  2 : Prev
  3 : Stop
'''

labels = [0,1,2,3]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import f1_score

PATH = "./Models/SimpleMusicModel_50.model"

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 6, kernel_size = 3,stride =2,padding =(0,0))
    self.bn2d_1 = nn.BatchNorm2d(6)
    self.pool1 = nn.MaxPool2d(kernel_size=2,stride = 2)
    self.conv2 = nn.Conv2d(in_channels = 6,out_channels = 10, kernel_size = 3,stride =2,padding =(0,0))
    self.bn2d_2 = nn.BatchNorm2d(10)
    self.pool2 = nn.MaxPool2d(kernel_size=2,stride = 2)
    self.fc = nn.Linear(40,60)
    self.bn1 = nn.BatchNorm1d(60)
    self.out_layer = nn.Linear(60, 4)
    
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


# class Net(nn.Module):
#   def __init__(self):
#     super(Net, self).__init__()
#     self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 16, kernel_size = 3,stride =2,padding =(0,0))
#     self.bn2d_1 = nn.BatchNorm2d(16)
#     self.pool1 = nn.MaxPool2d(kernel_size=2,stride = 1)
#     self.conv2 = nn.Conv2d(in_channels = 16,out_channels = 32, kernel_size = 3,stride =2,padding =(0,0))
#     self.bn2d_2 = nn.BatchNorm2d(32)
#     self.pool2 = nn.MaxPool2d(kernel_size=2,stride = 1)
#     self.fc = nn.Linear(3200,50)
#     self.bn1 = nn.BatchNorm1d(50)
#     self.out_layer = nn.Linear(50, 4)
    
#   def forward(self, x):
#     x = F.relu(self.conv1(x))
#     x = self.bn2d_1(x)
#     # print("After Conv1",x)
#     # x = self.bn1(x)
#     x = self.pool1(x)
#     # print("After Max pool1",x)
#     x = F.relu(self.conv2(x))
#     x = self.bn2d_2(x)
#     # x = self.bn2(x)
#     # print("After conv 2",x)
#     x = self.pool2(x)
#     # print("After Max pool 2",x)
#     x = x.view(-1, self.num_flat_features(x))
#     # print(x)
#     x = F.relu(self.fc(x))
#     # x = self.bn1(x)
#     # print("After Hideen Layer",x)
#     x = self.out_layer(x)
#     # print("After Output layer",x)
#     #x = F.log_softmax(x, dim=1)
#     x = F.softmax(x,dim=1)
#     # print(x)
#     return x

#   def num_flat_features(self, x):
#     size = x.size()[1:]  # all dimensions except the batch dimension
#     num_features = 1
#     for s in size:       # Get the products
#       num_features *= s
#     return num_features



net = Net()
net.load_state_dict(torch.load(PATH))

cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not cap.isOpened():
  raise IOError("Cannot open webcam")

classes = ["Background","Next","Prev","Stop"]

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
  ret, frame = cap.read()
  iframe = cv2.resize(frame, (50,50), interpolation = cv2.INTER_AREA)
  iframe = iframe/255.0
  inp = Variable(torch.from_numpy(iframe.reshape(1,iframe.shape[2],iframe.shape[0],iframe.shape[1])).float())
  # outputs = net(inp)
  outputs = F.softmax(net(inp),dim=1) # when using cross entropy as outputs are logits
  print(outputs)
  _, predicted = torch.max(outputs.data, 1)
  lab = classes[predicted]
  frame = cv2.putText(frame,lab,(140,250), font, .5,(255,2,5),2,cv2.LINE_AA)
  cv2.imshow('WebCam', frame)
  c = cv2.waitKey(1)
  if c == 27:
      break

cap.release()
cv2.destroyAllWindows()