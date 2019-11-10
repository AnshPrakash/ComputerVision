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

PATH = "./Models/SimpleMusicModel_5.model"

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
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img,3)
    edges = cv2.Canny(img,70,140)
    iedges = cv2.resize(edges, (50,50), interpolation = cv2.INTER_AREA)
    iedges = iedges/255.0
    iframe = iedges
    inp = Variable(torch.from_numpy(iframe.reshape(1,1,iframe.shape[0],iframe.shape[1])).float())
    outputs = net(inp)
    _, predicted = torch.max(outputs.data, 1)
    outputs = F.softmax(net(inp),dim=1) # when using cross entropy as outputs are logits
    print(outputs)
    lab = classes[predicted]
    frame = cv2.putText(frame,lab,(250,250), font, .9,(255,2,5),2,cv2.LINE_AA)
    cv2.imshow('WebCam', np.hstack([frame,cv2.merge([edges]*3)]))
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()