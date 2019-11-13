# e.g. python3 EstimateAccuracy.py ./Dataset train.mp4 no_of_epochs

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

from network import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = sys.argv[1] # Name of the folder containing dataset
typeVid = sys.argv[2] # expample for training train.mp4 ,for test it is test.mp4
graph_name = sys.argv[4]
labels = [0,1,2,3]
classes = [dataset +"/background/",dataset +"/next/",dataset +"/prev/",dataset +"/stop/"]
# for clss in classes:
#   print(clss+"trainn.mp4",count_frames(clss+"trainn.mp4"))



criterion = nn.CrossEntropyLoss()
Loss = []
iterations = []
Accuracy = []
# epochs = 100
epochs = int(sys.argv[3])
batch_size = 64

with torch.no_grad():
  for epoch_i in range(epochs):
    PATH = "./Models/SimpleMusicModel_"+str(epoch_i)+".model"
    net = Net().to(device)
    net.load_state_dict(torch.load(PATH, map_location=lambda storage, loc: storage))
    net.eval() 
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
        sizeOfBatch = 0
        batch_data = []
        batch_labels = []
        while sizeOfBatch!=batch_size and cap.isOpened():
          ret, frame = cap.read() 
          if ret== True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame/255.0 
            # frame = frame -frame.mean()
            batch_data.append(frame)
            batch_labels.append(label)
            sizeOfBatch += 1
          else:
            break  
        if (len(batch_labels)) != batch_size:
          break
        batch_data = np.array(batch_data)
        inputs = (torch.from_numpy(batch_data.reshape(batch_size,1,50,50)).float()).to(device)
        output = net(inputs)
        _, predicted = torch.max(output.data, 1)
        for i in range(len(batch_data)):
          conf[batch_labels[i]][predicted[i]] += 1
        loss = criterion(output,torch.from_numpy(np.array(batch_labels)).to(device)) #for cross entropy
        running_loss += loss.cpu().item()
        # print(output)
        # print(len(target))
        total_pts += len(batch_data)
      cap.release()
    print("Epoch ",epoch_i,"=>\n",np.array(conf)) 
    print("+++++++++++++++***********____________")
    Loss.append(running_loss/total_pts)
    Accuracy.append(np.trace(conf)/np.sum(conf))
    iterations.append(epoch_i)
    print("Accuracy ",np.trace(conf)/np.sum(conf))
    print("Runnning Loss on whole DataSet on epoch ",epoch_i," is ",running_loss/total_pts)
    


plt.ylabel('Loss')
plt.xlabel('epochs')
plt.plot(iterations, Loss, 'r')
plt.savefig(graph_name+"_set_graphs/"+ "Loss.png")
plt.clf()
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.plot(iterations, np.array(Accuracy)*100, 'b')
plt.savefig(graph_name+ "_set_graphs/" +"Accuracy.png")

