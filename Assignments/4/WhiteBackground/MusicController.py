from imutils.video import count_frames
import numpy as np
import cv2
import sys
import os
from network import *
from bgSub import process

'''
  0 : BackGround
  1 : Next
  2 : Prev
  3 : Stop
'''

labels = [0,1,2,3]

model_no = int(sys.argv[1])
PATH = "./Models/SimpleMusicModel_"+ str(model_no) +".model"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = Net().to(device)
net.load_state_dict(torch.load(PATH, map_location=lambda storage, loc: storage))
net.eval()

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(sys.argv[1])
# Check if the webcam is opened correctly
if not cap.isOpened():
  raise IOError("Cannot open webcam")

classes = ["Background","Next","Prev","Stop"]

font = cv2.FONT_HERSHEY_SIMPLEX


with torch.no_grad():
  while True:
    ret, frame = cap.read()
    img = cv2.resize(frame, (50,50), interpolation = cv2.INTER_AREA)
    img = process(img)
    img = img/255.0
    inp = Variable(torch.from_numpy(img.reshape(1,1,50,50)).float())
    outputs = net(inp)
    _, predicted = torch.max(outputs.data, 1)
    outputs = F.softmax(net(inp),dim=1) # when using cross entropy as outputs are logits
    print(outputs)
    lab = classes[predicted]
    frame = cv2.putText(frame,lab,(250,250), font, .9,(255,2,5),2,cv2.LINE_AA)
    cv2.imshow('WebCam', frame)
    cv2.imshow('DetectedSkin',img)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()