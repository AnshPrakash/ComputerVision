# Training with white background
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

def countFrames(vidFile):
  # count the total number of frames in the video file
  override = False 
  total = count_frames(vidFile, override=override)
  return(total)


classes = ["Dataset/background/","Dataset/next/","Dataset/prev/","Dataset/stop/"]
for clss in classes:
  print(clss+"train.mp4",count_frames(clss+"train.mp4"))

labels = [0,1,2,3]

# Pytorch Code Staring from here
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import f1_score



cuda_avail = torch.cuda.is_available()

def save_models(epoch):
  torch.save(net.state_dict(), "./Models/SimpleMusicModel_{}.model".format(epoch))
  print("Chekcpoint saved :",epoch)


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 16, kernel_size = 3,stride =2,padding =(0,0))
    self.bn2d_1 = nn.BatchNorm2d(16)
    self.pool1 = nn.MaxPool2d(kernel_size=2,stride = 1)
    self.conv2 = nn.Conv2d(in_channels = 16,out_channels = 32, kernel_size = 3,stride =2,padding =(0,0))
    self.bn2d_2 = nn.BatchNorm2d(32)
    self.pool2 = nn.MaxPool2d(kernel_size=2,stride = 1)
    self.fc = nn.Linear(3200,50)
    self.bn1 = nn.BatchNorm1d(50)
    self.out_layer = nn.Linear(50, 4)
    
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.bn2d_1(x)
    # print("After Conv1",x)
    # x = self.bn1(x)
    x = self.pool1(x)
    # print("After Max pool1",x)
    x = F.relu(self.conv2(x))
    x = self.bn2d_2(x)
    # x = self.bn2(x)
    # print("After conv 2",x)
    x = self.pool2(x)
    # print("After Max pool 2",x)
    x = x.view(-1, self.num_flat_features(x))
    # print(x)
    x = F.relu(self.fc(x))
    # x = self.bn1(x)
    # print("After Hideen Layer",x)
    x = self.out_layer(x)
    # print("After Output layer",x)
    #x = F.log_softmax(x, dim=1)
    x = F.softmax(x,dim=1)
    # print(x)
    return x

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:       # Get the products
      num_features *= s
    return num_features

net = Net().cuda() if cuda_avail else Net()
print(net)
print("Cuda ",cuda_avail)

# criterion = nn.NLLLoss()
# criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.01,momentum = 0.9,weight_decay=1e-5)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-8, amsgrad=False)
epochs = 100
batch_size = 32

Loss = []
iterations = []

for epoch_i in range(epochs):
  running_loss = 0.0
  total_pts = 0
  for label in labels:
    cap = cv2.VideoCapture(classes[label]+"train.mp4")
    if (cap.isOpened()== False):
      print("Error opening video file")
    one_hot = [0,0,0,0]
    one_hot[label] = 1
    while cap.isOpened():
      ret, frame = cap.read() 
      if ret == True:
        frame = cv2.resize(frame, (50,50), interpolation = cv2.INTER_AREA)
        frame = frame/255.0
        if cuda_avail:
          inputs = Variable(torch.from_numpy(frame.reshape(1,frame.shape[2],frame.shape[0],frame.shape[1]).float()).cuda())
          target = Variable(torch.from_numpy(np.array([one_hot]).float()).cuda())
        else:                
          inputs = Variable(torch.from_numpy(frame.reshape(1,frame.shape[2],frame.shape[0],frame.shape[1])).float())
          target = Variable(torch.from_numpy(np.array([one_hot])).float())

        optimizer.zero_grad()   # zero the gradient buffers
        output = net(inputs)
        # print(output)
        # print(target)
        loss = criterion(output, target)
        # loss = criterion(output)
        # print(loss.grad_fn )
        net.zero_grad()
        loss.backward()
        optimizer.step()    # Does the update
        running_loss += loss.cpu().item()
        total_pts += len(target)
      else:  
        break
    cap.release()
  save_models(epoch_i)
  Loss.append(running_loss/total_pts)
  iterations.append(epoch_i)
  print("Runnning Loss on whole DataSet on epoch ",epoch_i," is ",running_loss/total_pts)
  


plt.plot(iterations, Loss, 'r') # plotting t, a separately 
# plt.plot(iterations, b, 'ValidLoss') # plotting t, b separately 
plt.show()






# Save the Model
# torch.save(net.state_dict(),"BreakOutmodel.model")


# Cross Validation Set
# print("Checking Cross Validation Set")
# for i in range(151,200):
#   TrainingDataset = BreakOutDataset(file='TrainingData/'+str(i).zfill(8))
#   dataloader = DataLoader(TrainingDataset, batch_size=batch_size,shuffle=False, num_workers=4)
#   print("Accuracy ",accuracy('TrainingData/'+str(i).zfill(8)))
