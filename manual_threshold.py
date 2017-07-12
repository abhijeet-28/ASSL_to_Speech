import cv2
import numpy as np
import time
from math import sqrt


def thresh_det(x1, y1, x2, y2, img):
    averageB = 0
    averageG = 0
    averageR = 0
    sumB = 0
    sumG = 0
    sumR = 0

    averageB = np.average(img[x1:x2,y1:y2,0])
    averageG = np.average(img[x1:x2,y1:y2,1])
    averageR = np.average(img[x1:x2,y1:y2,2])
    varB, varG, varR = 0, 0,  0
    varB = sqrt(np.var(img[x1:x2,y1:y2,0]))
    varG = sqrt(np.var(img[x1:x2,y1:y2,1]))
    varR = sqrt(np.var(img[x1:x2,y1:y2,2]))
    factor = 0.8
    lowerb = np.array([averageB-varB*factor, averageG-varG*factor, averageR-varR*factor])
    upperb = np.array([averageB+varB*factor, averageG+varG*factor, averageR+varR*factor])
    return [cv2.inRange(img, lowerb, upperb), [averageB, averageG, averageR], [varB, varG, varR]]


def my_thresh(img, res, error):
    factor = 0.8
    lowerb = np.array([res[0]-error[0]*factor, res[1]-error[1]*factor, res[2]-error[2]*factor])
    upperb = np.array([res[0]+error[0]*factor, res[1]+error[1]*factor, res[2]+error[2]*factor])
    return cv2.inRange(img, lowerb, upperb)


   