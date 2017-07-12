import cv2
import numpy as np
import manual_threshold as th
import calibrate as cal
from matplotlib import pyplot as plt
import json
from threading import *

import csv



def parallel_fn(equY,count):
	equY_local=cal.transform(equY,colors_Y_Channel[colors[count]])

		


	equalisedYUV=cv2.merge([equY_local,imgYUV[:,:,1],imgYUV[:,:,2]])

	equalised_img=cv2.cvtColor(equalisedYUV, cv2.COLOR_YUV2BGR)

	cv2.imshow("equalised",im)
	

	out=th.my_thresh(equalised_img, colors_thresh[colors[count]][0], colors_thresh[colors[count]][1])

	thresholded=out
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
	dilated = cv2.dilate(thresholded, kernel)
	dilated = cv2.dilate(dilated, kernel)
	
	eroded = cv2.erode(dilated, kernel)
	eroded = cv2.erode(eroded, kernel)

	contours, hierarchy = cv2.findContours(eroded.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	if len(contours)!=0:
		pos = find_largest_contour(contours)
		contour = cv2.drawContours(im, contours, pos, colors_RGB[colors[count]], 3)

		cnt=contours[pos]
		M = cv2.moments(cnt)
		if M['m00']!=0:
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
		else:
			cx=None
			cy=None
		colors_centroid[colors[count]]=[cx,cy]

		cv2.circle(im, (cx,cy), 5, colors_RGB[colors[count]],1)	
	cv2.imshow('eroded '+str(colors[count]),eroded)




def find_largest_contour(contours):
    maxarea = 0
    pos = -1
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > maxarea:
            maxarea = area
            pos = i
    return pos


cap = cv2.VideoCapture(1)
# pool= multiprocessing.Pool(multiprocessing.cpu_count())

colors=["blue","yellow","green","orange","light_green"]
colors_RGB={"blue":(255,0,0),"yellow":(0,255,255),"green":(0,255,0),"purple":(128,0,128),"black":(0,0,0),"red":(0,0,255),"orange":(0,165,255),\
"light_green":(0,156,0)}
colors_centroid={"blue":[0,0],"yellow":[0,0],"green":[0,0],"orange":[0,0],"light_green":[0,0]}
command=raw_input("Do you want to use old calibrated values?(yes/no):")

if command=="yes":
	with open("calibrated_values"+'.txt') as json_file:
		preProcessedData=json.load(json_file)
	colors_thresh=preProcessedData["colors_thresh"]
	colors_Y_Channel=preProcessedData["colors_Y_Channel"]
else:
	colors_thresh,colors_Y_Channel=cal.calibrate(cap)


while 1:

	while 1:

		colors_centroid={"blue":[0,0],"yellow":[0,0],"green":[0,0],"orange":[0,0],"light_green":[0,0]}
		_, im = cap.read()
		
		imgYUV = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)

		equY = cv2.equalizeHist(imgYUV[:,:,0])

		count=0

		while count<5:


			# pool.apply_async( parallel_fn, (equY,count,) )

			t=Thread(target=parallel_fn, args=(equY,count,))
			
			t.start()
			t.join()
			count+=1





		cv2.imshow("image BGR", im)
		# cv2.imshow('threshold',thresholded)
		# cv2.imshow('dialated',dilated) 
		# cv2.imshow('eroded',eroded) 

		k = cv2.waitKey(5) & 0xFF
		if k==27:
			letter=raw_input("enter letter:")

			with open('data.csv', 'a') as f:

				wtr = csv.writer(f, delimiter= ',')
				temp=colors_centroid.values()
				for centroid in temp:
					data.append(centroid[0])
					data.append(centroid[1])

				data.append(letter)
				wtr.writerow(data)
			break


	k = cv2.waitKey(5) & 0xFF
	if k==27:

		print "exiting"
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