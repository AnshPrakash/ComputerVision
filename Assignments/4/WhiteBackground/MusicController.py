from imutils.video import count_frames
import numpy as np
import cv2
import sys
import os
from network import *

'''
  0 : BackGround
  1 : Next
  2 : Prev
  3 : Stop
'''

labels = [0,1,2,3]

PATH = "./Models/SimpleMusicModel_132.model"

net = Net()
net.load_state_dict(torch.load(PATH))
net.eval()

cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not cap.isOpened():
  raise IOError("Cannot open webcam")

classes = ["Background","Next","Prev","Stop"]

font = cv2.FONT_HERSHEY_SIMPLEX


with torch.no_grad():
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