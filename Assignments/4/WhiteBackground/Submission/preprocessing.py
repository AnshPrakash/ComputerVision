import cv2
import numpy as np
import imutils
import sys,os



fourcc = cv2.VideoWriter_fourcc(*'mp4v')

dataset = sys.argv[1] # just the name of folder and it should be in same directory as the code
typeVid = sys.argv[2] # train.mp4 or test.mp4

newDataset = "Edge"+dataset+"_"+typeVid[:-4]
try:
    os.mkdir(newDataset)
except OSError:
    print ("Creation of the directory %s failed" % newDataset)
    exit()
else:
    print ("Successfully created the directory %s " % newDataset)


classes = ["/background/","/next/","/prev/","/stop/"]
labels = [0,1,2,3]
for label in labels:
  cap = cv2.VideoCapture(dataset + classes[label]+typeVid)
  try:
    os.mkdir(newDataset+classes[label])
  except OSError:
      print ("Creation of the directory %s failed" % newDataset+classes[label])
      exit()
  else:
      print ("Successfully created the directory %s " % newDataset+classes[label])
  out = cv2.VideoWriter(newDataset + classes[label]+typeVid,fourcc,30,(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
  if (cap.isOpened()== False):
    print("Error opening video file")
  while cap.isOpened():
    ret, frame = cap.read() 
    if ret == True:
      img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      img = cv2.medianBlur(img,3)
      edges = cv2.Canny(img,70,140)
      edges = cv2.merge([edges]*3)
      out.write(edges)
    else:
      break
  cap.release()
  out.release()