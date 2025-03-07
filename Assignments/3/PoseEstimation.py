import cv2
import numpy as np
# import os,sys

chessX,chessY = 7 , 9

# Load Calibration Matrix
with np.load('CaliMat.npz') as X:
  mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

def draw(img, corners, imgpts):
  corner = tuple(corners[0].ravel())
  img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
  img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
  img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
  return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessY*chessX,3), np.float32)
objp[:,:2] = np.mgrid[0:chessX,0:chessY].T.reshape(-1,2)


axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

files = ["4.png"]

for fname in files:
  img = cv2.imread(fname)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  ret, corners = cv2.findChessboardCorners(gray, (chessX,chessY),None)
  print(ret)
  if ret == True:
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    # Find the rotation and translation vectors.
    _,rvecs, tvecs,_ = cv2.solvePnPRansac(objp, corners2, mtx, dist)

    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    img = draw(img,corners2,imgpts)
    cv2.imshow('img',img)
    k = cv2.waitKey(0)
    if k == 115:
      cv2.imwrite("pose.png", img)






