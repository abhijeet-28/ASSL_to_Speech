import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import json
import os 
import cv2



def load_data(downsampleBy,databases):
	X = np.empty(shape=[0, 480*640/(downsampleBy**2)])

	y=[]

	for i in range(1,databases+1):

		print "reading from "+'./db_new'+str(i)

		for fn in os.listdir('./db_new'+str(i)):

			print fn
			out = fn[0] 
			

			y_temp=np.zeros(25).tolist()

			y_temp[ord(out.upper())-65]=1
			y_temp1=ord(out.upper())-65
			y.append(y_temp)

			with open("./db_new"+str(i)+"/"+fn) as json_data:
				temp = json.load(json_data)

				key=temp.keys()

				temp= np.array(temp[key[0]])
				[m,n]=temp.shape
				temp = temp[0:m:downsampleBy,0:n:downsampleBy]
				[m,n]=temp.shape
				temp =np.reshape(temp,(1,m*n))
				X=np.append(X,temp,axis=0)


	y=np.array(y)

	return X,y

#X_global,y_global=load_data(4,4)
