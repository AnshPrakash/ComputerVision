import cv2
import numpy as np
import imutils
import sys


cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(sys.argv[1])


# dataset = sys.argv[1] # just the name of folder and it should be in same directory as the code
# typeVid = sys.argv[2] # train.mp4 or test.mp4
# newDataset = "Edge"+dataset+"_"+typeVid[:-4]
# classes = [dataset +"/background/",dataset +"/next/",dataset +"/prev/",dataset +"/stop/"]
# labels = [0,1,2,3]
# for label in labels;
#   cap = cv2.VideoCapture(classes[label]+typeVid)
#   if (cap.isOpened()== False):
#     print("Error opening video file")
#   while cap.isOpened():
#     ret, frame = cap.read() 
#     if ret == True:
#       img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#       cv2.medianBlur(img,3)
#       edges = cv2.Canny(img,70,140)
#       # cv2.imshow("WebCam", np.hstack([BlurImage(frame), edges]))
#       cv2.imshow('WebCam', edges)
#       c = cv2.waitKey(1)
#     else:
#       break
#   cap.release()

# out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output.avi",fourcc,30,(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while True:
  ret, frame = cap.read()
  if ret:
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.medianBlur(img,3)
    edges = cv2.Canny(img,70,140)
    out.write(edges)
    # cv2.imshow("WebCam", np.hstack([BlurImage(frame), edges]))
    cv2.imshow('WebCam', edges)
    c = cv2.waitKey(1)
    if c == 27:
        break
  else:
    break

cap.release()
out.release()
cv2.destroyAllWindows()



