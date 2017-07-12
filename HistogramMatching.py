import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image


def sampling(img):
    M = img.shape[0]
    N = img.shape[1]
    a = math.ceil(.1*math.sqrt(M*N/2.0) - 1)
    samples = []
    acc1 = 0
    # acc2 = 0
    while acc1 < M - 1 - a:
        acc2 = 0
        while acc2 < N - 1 - a:
            mat = img[acc1:acc1 + a, acc2: acc2+a]

            sample = center_value(mat)

            samples.append(sample)
            acc2 += a
        acc1 += a
    # samples.append([255,255,255])
    # print len(samples)
    return samples

def median_cut(colors):

    bucket = [colors]
    while(len(bucket) != 256):
        temp = []
        for i in range(0, len(bucket)):
            l1, l2 = median_cut_rec(bucket[i])
            temp.append(l1)
            temp.append(l2)
        bucket = []
        bucket.extend(temp)
    # return bucket
    samples =[]
    for color in bucket:
        sum1,sum2,sum3 = 0,0,0

        for i in range(0,len(color)):
            sum1 = sum1 + color[i][0]
            sum2 = sum2 + color[i][1]
            sum3 = sum3 + color[i][2]

            size = len(color)
            if size == 0 :
                break
        samples.append([sum3/size,sum2/size,sum1/size])
    return samples


def median_cut_rec(colors):
    idx = max_get_range(colors)
    colors = sorted(colors, key=lambda x: x[idx])
    mid = len(colors)/2
    return (colors[0:mid]), (colors[mid+1:len(colors)])

def max_get_range(colors):

    maxr, minr = max(colors,  key=lambda x: x[0]), min(colors,  key=lambda x: x[0])
    maxg, ming = max(colors,  key=lambda x: x[1]), min(colors,  key=lambda x: x[1])
    maxb, minb = max(colors,  key=lambda x: x[2]), min(colors,  key=lambda x: x[2])
    rangeR = maxr[0] - minr[0]
    rangeG = maxg[1] - ming[1]
    rangeB = maxb[2] - minb[2]
    maxRange = max(rangeR,rangeB,rangeG)
    if maxRange == rangeR:
        return 0
    if maxRange == rangeG:
        return 1
    if maxRange == rangeB:
        return 2



def center_value(mat):
    i = mat.shape[0]/2
    j = mat.shape[1]/2
    matl, mata, matb = cv2.split(mat)
    total = np.sum(matl)
    avg = total/float(mat.shape[0]*mat.shape[1] )
    # std = np.std(matl)
    lav = (.7) * matl[i,j] +(.3) * avg
    return (lav), mata[i,j], matb[i,j]

def histogram(img):
    # total = img.shape[0]*img.shape[1]
    histogram = {}
    for i in range(0,256):
        histogram[i] = 0.0;
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):

            histogram[round(img[i,j])] += 1.0

    return histogram

def histogramEquilization(histogram, total):
    histogramE = {}

    histogramE[0] = histogram[0]
    for i in range(1,256):
        histogramE[i] = histogramE[i-1] + histogram[i]
    minCDF = min(histogramE.values())
    for i in range(0, 256):
        histogramE[i] = int((histogramE[i] - minCDF) * 255 / (total - minCDF))
    return histogramE

def histogram_matching(histogram1, histogram2):
    newHistogram = {}
    inverseHisogram2 = {}
    for i in range(0,len(histogram2)):
        val1 = histogram2[0]
        for j in range(1, 256):
            val2 = histogram2[j]
            if i>=val1 and i<=val2:
                inverseHisogram2[i] = j
                break
            val1 = val2


    for i in range(0,len(histogram2)):
       newHistogram[i] = inverseHisogram2[histogram1[i]]

    return newHistogram

def transform(img, transformFunction):
    newimg = np.zeros(shape=img.shape, dtype=np.uint8)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            newimg[i,j] = transformFunction[round(img[i,j])]
    return newimg

def get_different_colors(img):
    colors = img.getcolors(img.size[0] * img.size[1])
    different_colors = [colors[i][1] for i in range(0, len(colors))]
    return different_colors


def main():

    target = cv2.imread('sample.jpg')
    source = cv2.imread('given.jpg')
    #target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)

    #
    #

    # # target = cv2.resize(target, None,fx = .5, fy = 0.5)
    s = source.copy()
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    # # cv2.imshow('p', target)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    print(source.shape)
    print(target.shape)
    # # cv2.imshow('pa', target)
    # cv2.waitKey()
    (l, a, b) = cv2.split(target)

    sl, sa, sb = cv2.split(source)
    # cv2.imshow('pa', sl)
    sl2 = sl.copy()
    # print sl,"\n", sa,"\n", sb
    mean_lsource = np.mean(sl)
    mean_ltarget = np.mean(l)
    std_lsource = np.std(sl)
    std_ltarget = np.std(l)
    sl = (std_ltarget / std_lsource) * (sl - mean_lsource) + mean_ltarget
    sl = np.clip(sl, 0, 255)
    sl = sl.astype(int)
    histogramSL = histogram(sl)
    histogramTL = histogram(l)
    histogram3 = histogram_matching(histogramEquilization(histogramSL, sl.shape[0]*sl.shape[1]), histogramEquilization(histogramTL, l.shape[0]*l.shape[1]))

    sl = transform(sl, histogram3)
    # cv2.imshow('ls',sl)


    # plt.bar(histogram2.keys(), histogram2.values(), 1, color='g')
    # plt.show()

    #
    src = cv2.merge([sl,sa,sb])
    src = cv2.cvtColor(src, cv2.COLOR_LAB2BGR)
    cv2.imshow('lumS', src)
    #
    # cv2.waitKey(0)
    # cv2.imwrite('../samples/test.jpg',src)
    # img = Image.open('../samples/test.jpg')
    #
    # colors = get_different_colors(img)
    # bucket = median_cut(colors)


    # sl = sl.astype(int)
    # print sl
    # histogram2 = histogram(sl)
    # plt.bar(histogram2.keys(), histogram2.values(), 1, color='g')
    # plt.show()


    # n = 1
    # p = [bucket[i:i + n] for i in range(0, len(bucket), n)]
    # g = np.asarray(p, dtype=np.uint8)
    # g = cv2.resize(g, (256,256), interpolation=cv2.INTER_LINEAR)
    # g = cv2.cvtColor(g, cv2.COLOR_BGR2LAB)
    # cv2.imshow('oo', g)
    # cv2.waitKey()
    # g = np.squeeze(g)
    # samples = sampling(source)

    # samples = g.tolist()
    # samples = sorted(samples, key=lambda x: x[0])
    # print samples
    #



    #newimg = cv2.cvtColor(newimg, cv2.COLOR_LAB2BGR)
    # newimg = cv2.resize(newimg, None, fx = 2, fy = 2)
    # cv2.imwrite('STD.jpg', newimg)
    # newimgl , newimga, newimgb = cv2.split(newimg)
    #cv2.imshow('pa', newimg)

    # # plt.show()

    cv2.imshow('original', s)
    target = cv2.cvtColor(target, cv2.COLOR_LAB2BGR)
    cv2.imshow('target', target)
    cv2.waitKey(0)



if __name__ == "__main__":
    main()
