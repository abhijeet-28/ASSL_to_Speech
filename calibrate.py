
import cv2
import numpy as np
import manual_threshold as th
import json

def calibrate(cap):

	colors_thresh={}
	colors_Y_Channel={}
	count=0

	while count<5:

		color_name=raw_input("Enter Color name:")
		while 1:

			_, im = cap.read()
			
			x1=200
			y1=200
			x2=x1+20
			y2=y1+20
			cv2.rectangle(im, (x1,y1), (x2, y2), (0, 255, 0), 2)

			imgYUV = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)

			equY = cv2.equalizeHist(imgYUV[:,:,0])

			hist,bins=np.histogram(equY.ravel(), 256,[0,256])


			cv2.imshow("image BGR", im)

			equalisedYUV=cv2.merge([equY,imgYUV[:,:,1],imgYUV[:,:,2]])

			im=cv2.cvtColor(equalisedYUV, cv2.COLOR_YUV2BGR)

			cv2.imshow("equalised",im)
			

			out=th.thresh_det(x1, y1, x2, y2, im)

			thresholded=out[0]
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
			dilated = cv2.dilate(thresholded, kernel)
			dilated = cv2.dilate(dilated, kernel)	
			eroded = cv2.erode(dilated, kernel)
			eroded = cv2.erode(eroded, kernel)

			
			cv2.imshow('threshold',thresholded)
			cv2.imshow('dialated',dilated) 
			cv2.imshow('eroded',eroded) 

			k = cv2.waitKey(5) & 0xFF
			if k==27:
				cv2.destroyAllWindows()				
				break

		colors_thresh[color_name]=[out[1],out[2]]
		colors_Y_Channel[color_name]=equalisedYUV[:,:,0].tolist()
		count+=1

	with open("calibrated_values"+'.txt', 'w') as outfile:
		outDict={"colors_thresh":colors_thresh,"colors_Y_Channel":colors_Y_Channel}
		json.dump(outDict, outfile,indent=4)

	return [colors_thresh, colors_Y_Channel]



def transform(sl,l):



    mean_lsource = np.mean(np.mean(sl))
    mean_ltarget = np.mean(l)
    std_lsource = np.std(sl)
    std_ltarget = np.std(l)

    sl = (std_ltarget / std_lsource) * (sl - mean_lsource) + mean_ltarget
    #sl = np.clip(sl, 0, 255)
    sl = sl.astype(np.uint8)
    #sl = np.float64(sl)

    return sl

