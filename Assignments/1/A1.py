import numpy as np
import cv2


def kers(size):
	return(np.ones((size,size),np.uint8))

cap = cv2.VideoCapture('Videos/8.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
count = 0
success, frame = cap.read()
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('video.mp4',fourcc,25,(frame.shape[1],frame.shape[2]))
while(success):
	fgmask = fgbg.apply(frame)
	dilation = cv2.dilate(fgmask,kers(2),iterations = 1)
	erosion = cv2.erode(dilation,kers(5),iterations = 2)
	dilation = cv2.dilate(erosion,kers(2),iterations = 2)
	# erosion = cv2.erode(dilation,kers(7),iterations = 2)
	# opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kers(2))
	# opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kers(10))
	# dilation = cv2.dilate(opening,kers(2),iterations = 2)
	# closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
	# closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	# opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
	edges = cv2.Canny(dilation,50,100,apertureSize = 3)
	# lines = cv2.HoughLines(edges,1,np.pi/180.0,100)
	lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength=10,maxLineGap=60)
	if(lines is not None):
		for line in lines:
			for x1,y1,x2,y2 in line:
				cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
	# out.write(frame)
	cv2.imshow('frame',frame)
	k = cv2.waitKey(1) & 0xff
	if k == 27:
		break
	count += 1
	success, frame = cap.read()

# out.release()		
cap.release()
cv2.destroyAllWindows()

