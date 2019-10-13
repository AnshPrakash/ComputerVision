import cv2

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
  raise IOError("Cannot open webcam")

captures = []
while True:
  ret, frame = cap.read()
  frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
  cv2.imshow('WebCam', frame)
  c = cv2.waitKey(1)
  if c == 99:
    captures.append(frame)
  if c == 27:
      break

num = 1
print(len(captures))
for img in captures:
  cv2.imwrite("./CameraCalPhotos/"+str(num)+".png",img)
  num += 1
cap.release()
cv2.destroyAllWindows()