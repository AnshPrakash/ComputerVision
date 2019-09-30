import cv2
import numpy as np
import sys
import os

SIZE = 1000
MIN_MATCH_COUNT = 10
GOOD_MATCH_PERCENT = 0.15
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)


# img1 gets projected to img2
def matchImages(img1,img2,kp1,kp2,des1,des2):
  matches = bf.match(des1,des2)
  matches.sort(key=lambda x: x.distance,reverse=False)
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  good = matches[:numGoodMatches]
  points1 = np.zeros((len(good), 2), dtype=np.float32)
  points2 = np.zeros((len(good), 2), dtype=np.float32)
  for i, match in enumerate(good):
    points1[i, :] = kp1[match.queryIdx].pt
    points2[i, :] = kp2[match.trainIdx].pt
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
  height, width, channels = img2.shape
  res =  np.concatenate((img2,np.zeros((height,width,3))),axis = 1)
  res = res.astype('uint8') 
  im1Reg = cv2.warpPerspective(img1, h, (width*2, height)) 
  b = (res == 0)
  res[b] = im1Reg[b]
  return res, h


images =  (os.listdir(sys.argv[1]))
images.sort()
kps = []

for imgf in range(images):
  print(os.path.join(os.getcwd(),os.path.join(sys.argv[1],imgf)))
  img = cv2.imread(os.path.join(os.getcwd(),os.path.join(sys.argv[1],imgf)))
  h,w,_ = img.shape
  nh,nw = SIZE*(h/max(h,w)),SIZE*(w/max(h,w))
  img = cv2.resize(img,(int(nh),int(nw)))
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  sift = cv2.xfeatures2d.SIFT_create()
  kp,des = sift.detectAndCompute(gray,None)
  kps.append((kp,des))
  # cv2.drawKeypoints(gray,kp,img)
  # cv2.imwrite('sift_keypoints.jpg',img)




idx1 = 1
idx2 = 0

i1 = cv2.imread(os.path.join(os.getcwd(),os.path.join(sys.argv[1],images[idx1])))
i2 = cv2.imread(os.path.join(os.getcwd(),os.path.join(sys.argv[1],images[idx2])))
h,w,_ = i1.shape
nh,nw = SIZE*(h/max(h,w)),SIZE*(w/max(h,w))
i1 = cv2.resize(i1,(int(nh),int(nw)))

h,w,_ = i2.shape
nh,nw = SIZE*(h/max(h,w)),SIZE*(w/max(h,w))
i2 = cv2.resize(i2,(int(nh),int(nw)))

res1 ,h = matchImages(i1,i2,kps[idx1][0],kps[idx2][0],kps[idx1][1],kps[idx2][1])
# idx1 = 0
# idx2 = 2
# res2 ,h = matchImages(i1,i2,kps[idx1][0],kps[idx2][0],kps[idx1][1],kps[idx2][1])


cv2.imshow("1",i1)
cv2.imshow("2",i2)


cv2.imshow("matching1", res1)
# cv2.imshow("matching2", res2)

cv2.waitKey(0)


####################################
# # for visualisation
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
# # Match descriptors.
# matches = bf.match(kps[0][1],kps[3][1])

# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
# # Draw first 10 matches.
# img_match = np.empty((max(i1.shape[0], i2.shape[0]), i1.shape[1] + i2.shape[1], 3), dtype=np.uint8)
# cv2.drawMatches(i1,kps[3][0],i2,kps[0][0],matches[:10],img_match,matchColor=None, singlePointColor=(255, 255, 255), flags=2)

# img_match =cv2.resize(img_match,(1000,500))
# cv2.imshow("BF matching", img_match)
# cv2.waitKey(0)
################################################################

