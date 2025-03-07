import numpy as np
import cv2
import sys
import os

chessX,chessY = 6 , 9

# termination criteria

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessX*chessY,3), np.float32)
objp[:,:2] = np.mgrid[0:chessX,0:chessY].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


images = (os.listdir(sys.argv[1]))


for fname in images:
  img = cv2.imread(os.path.join(os.getcwd(),os.path.join(sys.argv[1],fname)))
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  ret, corners = cv2.findChessboardCorners(gray, (chessX,chessY),None)
  if ret == True:
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners2)
    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, (chessX,chessY), corners2,ret)
    # cv2.imwrite("./ChessBoardCorners/"+str(fname),img)
    cv2.imshow('img',img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


np.savez('CaliMat.npz', mtx = mtx, dist =dist, rvecs = rvecs,tvecs = tvecs)



# img = cv2.imread('4.png')
# h, w = img.shape[0],img.shape[1]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# # undistort
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# # crop the image
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult.png',dst)

