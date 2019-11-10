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

classes = ["/background/","/next/","/prev/","/stop/"]
classes = [dataset +c for c in classes]
for clss in classes:
  print(clss+"train.mp4",count_frames(clss+"train.mp4"))

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
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
epochs = 200
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
      sizeOfBatch = 0
      batch_data = []
      batch_labels = []
      while sizeOfBatch!=batch_size and cap.isOpened():
        ret, frame = cap.read() 
        if ret== True:
          frame = cv2.split(frame)[0]
          frame = cv2.resize(frame, (50,50), interpolation = cv2.INTER_AREA)
          frame = frame/255.0 
          batch_data.append(frame)
          batch_labels.append(label)
          sizeOfBatch += 1
        else:
          break  
      if (len(batch_labels)) != batch_size:
        break
      batch_data = np.array(batch_data)
    
      # ret, frame = cap.read() 
      # frame = cv2.split(frame)[0]
      # frame = cv2.resize(frame, (50,50), interpolation = cv2.INTER_AREA)
      # frame = frame/255.0
      # inputs = (torch.from_numpy(frame.reshape(batch_size,1,frame.shape[0],frame.shape[1])).float()).to(device)
      # target = (torch.from_numpy(np.array([one_hot])).float()).to(device)

      inputs = (torch.from_numpy(batch_data.reshape(batch_size,1,50,50)).float()).to(device)
      output = net(inputs)
      # print(output)
      # print(output.size())
      # print(batch_labels)
      # print(target)
      # loss = criterion(output, target)
      loss = criterion(output, torch.from_numpy(np.array(batch_labels)).to(device)) #for cross entropy
      optimizer.zero_grad()   # zero the gradient buffers
      net.zero_grad()
      loss.backward()
      optimizer.step()
      running_loss += loss.cpu().item()
      total_pts += len(batch_labels)
    cap.release()
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
