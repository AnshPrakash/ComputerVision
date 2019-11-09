# e.g. python3 EstimateAccuracy.py ./Dataset train.mp4

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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import f1_score
from network import *


dataset = sys.argv[1] # Name of the folder containing dataset
typeVid = sys.argv[2] # expample for training train.mp4 ,for test it is test.mp4
labels = [0,1,2,3]
classes = [dataset +"/background/",dataset +"/next/",dataset +"/prev/",dataset +"/stop/"]

criterion = nn.CrossEntropyLoss()
Loss = []
iterations = []
Accuracy = []
epochs = 2
with torch.no_grad():
  for epoch_i in range(epochs):
    PATH = "./Models/SimpleMusicModel_"+str(epoch_i)+".model"
    net = Net()
    net.load_state_dict(torch.load(PATH))
    running_loss = 0.0
    total_pts = 0
    conf = [[0]*4 for _ in range(4)]
    for label in labels:
      cap = cv2.VideoCapture(classes[label]+typeVid)
      if (cap.isOpened()== False):
        print("Error opening video file")
      # one_hot = [0,0,0,0]
      # one_hot[label] = 1
      while cap.isOpened():
        ret, frame = cap.read() 
        if ret == True:
          frame = cv2.resize(frame, (50,50), interpolation = cv2.INTER_AREA)
          frame = frame/255.0
          inputs = Variable(torch.from_numpy(frame.reshape(1,frame.shape[2],frame.shape[0],frame.shape[1])).float())
          # target = Variable(torch.from_numpy(np.array([one_hot])).float())
          output = net(inputs)
          # print(output)
          # print(target)
          # loss = criterion(output, target)
          _, predicted = torch.max(output.data, 1)
          conf[label][predicted] += 1
          loss = criterion(output, torch.from_numpy(np.array([label]))) #for cross entropy
          running_loss += loss.item()
          # print(len(target))
          total_pts += 1
        else:
          break
      cap.release()
    print("Epoch ",epoch_i,"=>\n",np.array(conf)) 
    print("+++++++++++++++***********____________")
    Loss.append(running_loss/total_pts)
    Accuracy.append(np.trace(conf)/np.sum(conf))
    iterations.append(epoch_i)
    print("Runnning Loss on whole DataSet on epoch ",epoch_i," is ",running_loss/total_pts)
    


# plt.ylabel('Loss')
plt.xlabel('epochs')
plt.plot(iterations, Loss, 'r')
plt.plot(iterations, Accuracy, 'b')
plt.savefig('Graph.png')


