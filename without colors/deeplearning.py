import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import json
import os 
import cv2
import load_data

from keras.layers import Input, Dense
from keras.models import Model
import keras
from keras.datasets import mnist
from keras import regularizers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
import csv
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import copy as cp
import load_data
import reduce_features

downsampleBy=4
number_of_features=500

# X_global,y_global=load_data.load_data(downsampleBy,4)

#X_global_reduce,y_global_reduce= reduce_features.reduce_features(X_global,y_global,number_of_features)


X=X_global_reduce
y=y_global_reduce

print X.shape
print y.shape



x_train=X/255
y_train=y

x_test=x_train
y_test=y_train

sampleSet=[x for x in range(x_train.shape[0])]
cols_test=[np.random.randint(x_train.shape[0]) for x in range(int(x_train.shape[0]/3))]
cols_train=list(set(sampleSet)-set(cols_test))

x_test=x_train[cols_test,:]
x_train=x_train[cols_train,:]

y_test=y_train[cols_test,:]
y_train=y_train[cols_train,:]

'''
# y_test=y_train[cols_test]
# y_train=y_train[cols_train]

# clf = LinearDiscriminantAnalysis()
# clf.fit(x_train, y_train)

# accuracy=clf.score(x_test,y_test)

# print accuracy'''
# # this is our input placeholder
input_img = Input(shape=(x_train.shape[1],))
encoded = Dense(256, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
# encoded = Dense(encoding_dim, activation='relu',
#                  activity_regularizer=regularizers.activity_l1(10e-5))(input_img)

decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(x_train.shape[1], activation='relu')(decoded)

# # this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)

####################################################################

##########################################################################################
autoencoder.compile(optimizer='adadelta', loss='mse',metrics=['mae', 'acc'])


autoencoder.fit(x_train, x_train,
                nb_epoch=200,
                batch_size=100,
                shuffle=True,
                validation_data=(x_test, x_test))

##################################################################################################



# ##############################classifier########################################


classifier_layer1=Dense(128, activation='relu')(encoded)
classifier_layer2=Dense(128, activation='relu')(classifier_layer1)
classifier_layer3=Dense(y.shape[1], activation='softmax')(classifier_layer1)


SAE_classifier = Model(input=input_img, output=classifier_layer3)


SAE_classifier.compile(optimizer='adadelta', loss='mse',metrics=['mae', 'acc'])


SAE_classifier.fit(x_train, y_train,
                nb_epoch=500,
                batch_size=100,
                shuffle=True,
                validation_data=(x_test, y_test))

predicted=SAE_classifier.predict(x_test)


SAE_classifier.save("./models/firstModeregularise"+'.h5')

with open('y_pred.csv', 'w') as f:
    wtr = csv.writer(f, delimiter= ',')
    for i in range(len(predicted)):
        wtr.writerow(predicted[i])

with open('y_true.csv', 'w') as f:
    wtr = csv.writer(f, delimiter= ',')
    for i in range(len(predicted)):
        wtr.writerow(y_test[i])


y_true_conf_input=[]

for i in range(len(y_test)):
    y_true_conf_input.append(np.argmax(y_test[i]))


y_pred_conf_input=[]

for i in range(len(y_test)):
    y_pred_conf_input.append(np.argmax(predicted[i]))




confusion_mat=confusion_matrix(y_true_conf_input,y_pred_conf_input)
print(confusion_mat)

print(classification_report(y_true_conf_input, y_pred_conf_input))
