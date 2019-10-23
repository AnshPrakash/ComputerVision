from ModelLoader import *
from matplotlib import pyplot as plt
import sys,os
import cv2
import numpy as np
from cv2 import aruco


sift = cv2.xfeatures2d.SIFT_create()
MIN_MATCH_COUNT = 10
GOOD_MATCH_PERCENT = 0.20
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

def getHomograpy(img1,img2):
  gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
  kp1,des1 = sift.detectAndCompute(gray1,None)
  kp2,des2 = sift.detectAndCompute(gray2,None)
  matches = bf.match(des2,des1)
  matches.sort(key=lambda x: x.distance,reverse=False)
  if len(matches) < MIN_MATCH_COUNT :
    return([False,None])
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  good = matches[:numGoodMatches]
  points1 = np.zeros((len(good), 2), dtype=np.float32)
  points2 = np.zeros((len(good), 2), dtype=np.float32)
  for i, match in enumerate(good):
    points2[i, :] = kp2[match.queryIdx].pt
    points1[i, :] = kp1[match.trainIdx].pt
  h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
  # matchesMask = mask.ravel().tolist()
  # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
  #                  singlePointColor = None,
  #                  matchesMask = matchesMask[:10], # draw only inliers
  #                  flags = 2)
  # img3 = cv2.drawMatches(img2,kp2,img1,kp1,good[:10],None,**draw_params)
  # plt.imshow(img3, 'gray'),plt.show()
  return([True,h])

# Just like that
# img1 = cv2.imread(sys.argv[1])
# img2 = cv2.imread(sys.argv[1])
# H = getHomograpy(img2,img1)

# Load Calibration Matrix
with np.load('CaliMat.npz') as X:
  mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]


aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()

marker1 = cv2.imread("./Markers/marker1.png",cv2.IMREAD_GRAYSCALE)
marker1_pts, markerId1, _ = aruco.detectMarkers(marker1, aruco_dict, parameters=parameters)
marker1_pts = np.array(marker1_pts[0][0])
z = np.zeros((marker1_pts.shape[0],1))
marker1_pts = np.append(marker1_pts, z, axis=1)
marker1_pts = marker1_pts[:, np.newaxis, :]


marker2 = cv2.imread("./Markers/marker2.png",cv2.IMREAD_GRAYSCALE)
marker2_pts, markerId2, _ = aruco.detectMarkers(marker2, aruco_dict, parameters=parameters)
marker2_pts = np.array(marker2_pts[0][0])
z = np.zeros((marker2_pts.shape[0],1))
marker2_pts = np.append(marker2_pts, z, axis=1)
marker2_pts = marker2_pts[:, np.newaxis, :]




s = ""
l = (sys.argv[1].split("/"))
for fil in l[:-1]:
  s = s + fil +"/"

os.chdir(s)
print(os.getcwd())

obj = OBJ(l[-1], swapyz=True)
obj.vertices = np.array(obj.vertices)
# print(obj.normals)
# print(obj.texcoords)
# print(obj.faces)

markerLength = 1.2


def motionVector(v1,v2,steps):
  return((v2 - v1)/steps)

def render(frame,rvecs, tvecs, mtx, dist,tx,ty):
  vertices = obj.vertices
  scale_matrix = np.eye(3) * -3
  trans = np.array([tx,ty])
  h, w = marker1.shape
  for face in obj.faces:
    face_vertices = face[0]
    points = np.array([vertices[vertex - 1] for vertex in face_vertices])
    points = np.dot(points, scale_matrix)
    points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
    imgpts, jac = cv2.projectPoints(points.reshape(-1, 1, 3), rvecs, tvecs, mtx, dist)
    imgpts = imgpts + trans
    imgpts = np.int32(imgpts)
    frame = cv2.fillConvexPoly(frame, imgpts, (137, 27, 211))
  return(frame)


cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not cap.isOpened():
  raise IOError("Cannot open webcam")


tx,ty = 0,0
steps = 10
while True:
  ret, frame = cap.read()
  frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
  h, w = frame.shape[0],frame.shape[1]
  newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
  frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  parameters =  aruco.DetectorParameters_create()
  corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
  if ids is not None and len(ids)>1:
    if ids[0] != markerId1:
      # print("Before",corners)
      temp = corners[0].copy()
      corners[0] = corners[1].copy()
      corners[1] = temp
      temp = ids[0].copy()
      ids[0] = ids[1].copy()
      ids[1] = temp
      # print("After",corners)

  if ids is not None and ids[0] == markerId1:
    corners1 = np.array(corners[0][0])
    corners1 = corners1[:, np.newaxis, :]
    _,rvecs, tvecs,_ = cv2.solvePnPRansac(marker1_pts, corners1, mtx, dist)
    render(frame,rvecs, tvecs, mtx, dist,tx,ty)
  if ids is not None and ids.shape[0]>1 and ids[1] == markerId2:
    v1 = (np.mean(corners[0][0],axis = 0))
    v2 = (np.mean(corners[1][0],axis = 0))
    v1[0] = v1[0] + tx
    v1[1] = v1[1] + ty
    dx,dy = motionVector(v1,v2,steps)
    tx = tx + dx
    ty = ty + dy
    # print("dx,dy",dx,dy)
  frame = aruco.drawDetectedMarkers(frame, corners, ids)
  cv2.imshow('WebCam', frame)
  c = cv2.waitKey(1)
  if c == 27:
      break

cap.release()
cv2.destroyAllWindows()