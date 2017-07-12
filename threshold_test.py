import cv2
import numpy as np
import manual_threshold as th
import calibrate as cal
from matplotlib import pyplot as plt

def find_largest_contour(contours):
    maxarea = 0
    pos = -1
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > maxarea:
            maxarea = area
            pos = i
    return pos


cap = cv2.VideoCapture(0)


colors_thresh,colors_Y_Channel=cal.calibrate(cap)

while 1:
	_, im = cap.read()
	
	imgYUV = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)

	equY = cv2.equalizeHist(imgYUV[:,:,0])

	hist,bins=np.histogram(equY.ravel(), 256,[0,256])

	equY=cal.transform(equY,colors_Y_Channel["Blue"])

	


	equalisedYUV=cv2.merge([equY,imgYUV[:,:,1],imgYUV[:,:,2]])

	im=cv2.cvtColor(equalisedYUV, cv2.COLOR_YUV2BGR)

	cv2.imshow("equalised",im)
	

	out=th.my_thresh(im, colors_thresh["Blue"][0], colors_thresh["Blue"][1])

	thresholded=out
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
	dilated = cv2.dilate(thresholded, kernel)
	dilated = cv2.dilate(dilated, kernel)
	dilated = cv2.dilate(dilated, kernel)	
	eroded = cv2.erode(dilated, kernel)
	eroded = cv2.erode(eroded, kernel)

	contours, hierarchy = cv2.findContours(eroded.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	if len(contours)!=0:
		pos = find_largest_contour(contours)
		contour = cv2.drawContours(im, contours, pos, (0, 0, 255), 3)


	cv2.imshow("image BGR", im)
	cv2.imshow('threshold',thresholded)
	cv2.imshow('dialated',dilated) 
	cv2.imshow('eroded',eroded) 

	k = cv2.waitKey(5) & 0xFF
	if k==27:
		break

# while(1):

# 	_, im = cap.read()
# 	cv2.imshow("image", im)

# 	# r = cv2.getTrackbarPos('R','image')
# 	# g = cv2.getTrackbarPos('G','image')
# 	# b = cv2.getTrackbarPos('B','image')
# 	# varience=20
# 	# lower_blue = np.array([b-varience,g-varience,r-varience]) #54 89 238
# 	# upper_blue = np.array([b+varience,g+varience,r+varience])
# 	# mask = cv2.inRange(im, lower_blue, upper_blue)

# 	thresh_det(x1, y1, x2, y2, img)

# 	cv2.imshow('threshold',mask)

# 	k = cv2.waitKey(5) & 0xFF

# 	if k==27:
# 		break

# cv2.destroyAllWindows()