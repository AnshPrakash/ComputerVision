import numpy as np
import cv2
import sys,os

def process(img):
  hue_range = (0, 20)
  lightness_range = (30, 150)
  saturation_range = (0, 150)
  blur_size = (5,5)
  img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
  _lower_hls = np.array([hue_range[0], lightness_range[0], saturation_range[0]])
  _upper_hls = np.array([hue_range[1], lightness_range[1], saturation_range[1]])
  img_mask = cv2.inRange(img_hls, _lower_hls, _upper_hls)
  img_blur = cv2.GaussianBlur(img_mask, blur_size, 0)
  _, img_res = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  img_res = cv2.resize(img_res,(50,50))
  # cv2.imshow("output", img_res)
  # cv2.waitKey()
  return img_res

# folder = sys.argv[1] # fodler name
# file = sys.argv[2] # file name

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# newfile = os.path.join(folder,"train_bg.mp4")
# out = cv2.VideoWriter(newfile,fourcc,30,(int(50),int(50)))

# cap = cv2.VideoCapture(os.path.join(folder,file))
# if not cap.isOpened():
#   raise IOError("Cannot open")
# while True:
#   ret, frame = cap.read()
#   if ret == True:
#     img = cv2.resize(frame, (50,50), interpolation = cv2.INTER_AREA)
#     img = process(img)
#     out.write(cv2.merge([img,img,img]))
#     cv2.imshow('WebCam', img)
#     c = cv2.waitKey(1)
#     if c == 27:
#         break
#   else:
#     break

# cap.release()
# out.release()
# cv2.destroyAllWindows()