import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
import csv
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import copy as cp


def reduce_features(X_global,y_global,number_of_features):
	X=cp.deepcopy(X_global)

	print "reducing dimensionality"
	pca = PCA(n_components=number_of_features)

	X_global_reduce=pca.fit_transform(X)	

	print "saving  PCA model"
	pickle.dump(pca, open("./models/pcafirstModel1000", 'wb'))

	return X_global_reduce , y_global

#X_global_reduce,y_global_reduce=reduce_features(X_global,y_global)