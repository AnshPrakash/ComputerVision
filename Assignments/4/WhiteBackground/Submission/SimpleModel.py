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

dataset = sys.argv[1] 
typeVid = sys.argv[2] # example handDetected(train_bg.mp4)  or raw(train.mp4)

classes = ["/background/","/next/","/prev/","/stop/"]
classes = [dataset +c for c in classes]
for clss in classes:
  print(clss+"train.mp4",count_frames(clss+typeVid))

labels = [0,1,2,3]

# Pytorch Code Staring from here
from network import *


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def save_models(epoch):
  torch.save(net.state_dict(), "./Models/SimpleMusicModel_{}.model".format(epoch))
  print("Chekcpoint saved :",epoch)


torch.manual_seed(0)
net = Net().to(device)
print(net)

# criterion = nn.NLLLoss()
# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001,momentum = 0.9,weight_decay=1e-5)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001,weight_decay=1e-4)
epochs = 100
batch_size = 64

Loss = []
iterations = []

net.train()
for epoch_i in range(epochs):
  caps = []
  for label in labels:
    caps.append(cv2.VideoCapture(classes[label]+typeVid))
    if (caps[label].isOpened()== False):
      print("Error opening video file")
      exit()
  running_loss = 0.0
  total_pts = 0
  while caps[0].isOpened() or caps[1].isOpened() or caps[2].isOpened() or caps[3].isOpened():
    sizeOfBatch = 0
    batch_data = []
    batch_labels = []
    while (caps[0].isOpened() or caps[1].isOpened() or caps[2].isOpened() or caps[3].isOpened()) and sizeOfBatch!=batch_size:
      for label in labels:
        if sizeOfBatch == batch_size:
          break
        ret, frame = caps[label].read() 
        # print(ret,sizeOfBatch)
        if ret== True:
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          frame = frame/255.0
          # frame = frame -frame.mean()
          batch_data.append(frame)
          batch_labels.append(label)
          sizeOfBatch += 1
        else:
          caps[label].release()
          continue
    if (len(batch_labels)) != batch_size:
      break
    # print("Got data",sizeOfBatch)
    batch_data = np.array(batch_data)
    inputs = Variable(torch.from_numpy(batch_data.reshape(batch_size,1,50,50)).float()).to(device)
    output = net(inputs)
    loss = criterion(output, Variable(torch.from_numpy(np.array(batch_labels))).to(device)) #for cross entropy
    optimizer.zero_grad()   # zero the gradient buffers
    net.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss += loss.cpu().item()
    total_pts += len(batch_labels)
  for label in labels:
    caps[label].release()
  save_models(epoch_i)
  Loss.append(running_loss/total_pts)
  iterations.append(epoch_i)
  print("Runnning Loss on whole DataSet on epoch ",epoch_i," is ",running_loss/total_pts)
  

plt.ylabel('Loss')
plt.xlabel('epochs')
plt.plot(iterations, Loss, 'r') # plotting t, a separately 
# plt.plot(iterations, b, 'ValidLoss') # plotting t, b separately 
plt.savefig('TrainLoss.png')
plt.show()





# Save the Model
# torch.save(net.state_dict(),"BreakOutmodel.model")


# Cross Validation Set
# print("Checking Cross Validation Set")
# for i in range(151,200):
#   TrainingDataset = BreakOutDataset(file='TrainingData/'+str(i).zfill(8))
#   dataloader = DataLoader(TrainingDataset, batch_size=batch_size,shuffle=False, num_workers=4)
#   print("Accuracy ",accuracy('TrainingData/'+str(i).zfill(8)))
