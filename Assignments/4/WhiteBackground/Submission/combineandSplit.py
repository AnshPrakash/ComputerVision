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




files = [os.path.join(sys.argv[1], f) for f in os.listdir(sys.argv[1]) if os.path.isfile(os.path.join(sys.argv[1], f))]

caps = []
for f in files:
  caps.append(cv2.VideoCapture(f))
 
for cap in caps:
  print(cap.isOpened())

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

newfile1 = sys.argv[1]+"/train.mp4"
newfile2 = sys.argv[1]+"/test.mp4"
out1 = cv2.VideoWriter(newfile1,fourcc,30,(int(50),int(50)))
out2 = cv2.VideoWriter(newfile2,fourcc,30,(int(50),int(50)))

for cap in caps:
  while True:
    ret, frame = cap.read()
    if ret == True:
      frame = cv2.resize(frame, (50,50), interpolation = cv2.INTER_AREA)
      if np.random.uniform(0,1) <= 0.8:
        out1.write(frame)
      else:
        out2.write(frame)
    else:
      cap.release()
      break

out1.release()
out2.release()