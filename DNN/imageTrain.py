'''

~~~ IDEAS ~~~

1. Use ImageAugmentation from Keras
https://machinelearningmastery.com/image-augmentation-deep-learning-keras/

	This allows to randomly change images with rotation, translation, whitening, brightening, etc.
	It seems that CNN are NOT rotation invariant. 
	https://stackoverflow.com/questions/41069903/why-rotation-invariant-neural-networks-are-not-used-in-winners-of-the-popular-co
	Image augmentation can help. But I have already tons of events coming from all directions
		-> Is Image Augmentation needed?
		
2. Use Inception v1/v2 modules
https://towardsdatascience.com/neural-network-architectures-156e5bad51ba
	
	The above article makes it sound logical and powerful. Adapted to my case? Overkill? 
	Instead of doing the full GoogLeNet, can use only 2-3 Inception modules ?
	
3. Use ResNet modules / Inception v4
https://towardsdatascience.com/neural-network-architectures-156e5bad51ba
https://github.com/raghakot/keras-resnet

	Have to find more literature on ResNet. Idea sounds simple. Actual implementation ... ?
	
4. Use two 7x22 images instead of one 14x22 ?

	Shuffling becomes even harder...
	Feed images in parallel then use Concatenate() 
	... I would really need opinions from an expert on this ...
	
5. Reprocess images to align XZ and YZ?

	For example: for every YZ layer, compute the COG and shift it in-between the COG of layers +1 and -1
	... Would lose 3D information ?
	... Would make problems due to limited resolution? (what if COG falls in-between two pixels?)
	
6. Feed variables in parallel to the image?

	What kind of variables ?
	
7. Hyperparameter optimization - grid search

	Sounds really costly and intensive to do that. Especially if the above is not yet decided/tested/under control
	
	But some of these parameters can affect convergence ? ... Or not ?
	
	
8. If I go for parallel architecture, how do I remove the output sigmoid ? Can't simply iterate on layers to transfer weights?

'''


from __future__ import print_function, division, absolute_import

import numpy as np
import time
import pickle
import sys
import os
import glob
from uncertainties import ufloat
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Keras deep neural networks
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, ReduceLROnPlateau
from keras import regularizers
from keras.optimizers import Adam


def getModel(X_train):
	'''
	https://www.kaggle.com/adityaecdrid/mnist-with-keras-for-beginners-99457/notebook
	
	'''
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=(14,22,1)))
	model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
	#https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.20))
	model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
	model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))
	model.add(Dense(1, activation='sigmoid'))
	
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
	
	return model
	
def getLinearModel(X_train,model):
	model2 = Sequential()
	model2.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=(14,22,1)))
	model2.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
	#https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
	model2.add(MaxPooling2D((2, 2)))
	model2.add(Dropout(0.20))
	model2.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
	model2.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
	model2.add(MaxPooling2D(pool_size=(2, 2)))
	model2.add(Dropout(0.25))
	model2.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
	model2.add(Dropout(0.25))
	model2.add(Flatten())
	model2.add(Dense(128, activation='relu'))
	model2.add(BatchNormalization())
	model2.add(Dropout(0.25))
	model2.add(Dense(1))
	
	for i,x in enumerate(model.layers):
		weights = x.get_weights()
		model2.layers[i].set_weights(weights)
		
	return model2	
	
	
def train():
	
	PATH = '/home/drozd/analysis/runs/run_29Jan19_image/inFiles/'
	
	arr_p = np.load(PATH+'bigArr_32b_p.npy')
	arr_e = np.load(PATH+'bigArr_32b_e.npy')
	
	np.random.shuffle(arr_e)
	np.random.shuffle(arr_p)
	
	if arr_p.shape[0] > arr_e.shape[0]:
		arr_p = arr_p[ :arr_e.shape[0], ]
	else:
		arr_e = arr_e[ :arr_p.shape[0], ]
	
	print("arr_e:",arr_e.shape)
	print("arr_p:",arr_p.shape)
	
	X = np.concatenate(( arr_e,arr_p ))
	y = np.concatenate(( np.ones( (arr_e.shape[0],) ), np.zeros( (arr_p.shape[0],) ) ))
	
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,shuffle=True)
	
	X_train = X_train.reshape( (X_train.shape[0],X_train.shape[1],X_train.shape[2],1) )
	X_test = X_test.reshape( (X_test.shape[0],X_test.shape[1],X_test.shape[2],1) )
	
	model = getModel(X_train)
	rdlronplt = ReduceLROnPlateau(monitor='loss',patience=3,verbose=1,factor=0.5,min_lr=0.0001)
	modelcheckpoint = ModelCheckpoint('out/weights.{epoch:02d}-{val_loss:.4f}.hdf5',save_weights_only=True)
	
	history = model.fit(X_train,y_train,batch_size=100,epochs=150,verbose=2,callbacks=[rdlronplt,modelcheckpoint],validation_data=(X_test,y_test))
	
	predictions = model.predict(X_test)
	elecs_p = predictions[ y_test.astype(bool) ]
	prots_p = predictions[ ~y_test.astype(bool) ]
	
	fig1 = plt.figure()
	plt.plot(history.history['loss'],label='loss')
	plt.plot(history.history['val_loss'],label='val_loss')
	plt.legend(loc='best')
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.title('train history')
	plt.savefig('plots/history')
	plt.yscale('log')
	plt.savefig('plots/history_log')
	plt.close(fig1)
	
	fig2 = plt.figure()
	binList = np.linspace(0,1,100)
	plt.hist(elecs_p,bins=binList,label='e',histtype='step',color='green')
	plt.hist(prots_p,bins=binList,label='p',histtype='step',color='red')
	plt.ylim(ymin=0.9)
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.legend(loc='best')
	plt.grid(True)
	plt.yscale('log')
	plt.savefig('plots/classScore_sigmoid')
	plt.close(fig2)
	
	
	model2 = getLinearModel(X_train,model)
	model2.save('out/model.h5')
	predictions = model2.predict(X_test)
	elecs_p = predictions[ y_test.astype(bool) ]
	prots_p = predictions[ ~y_test.astype(bool) ]
	fig2 = plt.figure()
	binList = np.linspace(np.min(predictions),np.max(predictions),200)
	plt.hist(elecs_p,bins=binList,label='e',histtype='step',color='green')
	plt.hist(prots_p,bins=binList,label='p',histtype='step',color='red')
	plt.ylim(ymin=0.9)
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.legend(loc='best')
	plt.grid(True)
	plt.yscale('log')
	plt.savefig('plots/classScore_linear_wide')
	plt.close(fig2)
	
	fig2 = plt.figure()
	binList = np.linspace(-10,10,100)
	plt.hist(elecs_p,bins=binList,label='e',histtype='step',color='green')
	plt.hist(prots_p,bins=binList,label='p',histtype='step',color='red')
	plt.ylim(ymin=0.9)
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.legend(loc='best')
	plt.grid(True)
	plt.yscale('log')
	plt.savefig('plots/classScore_linear')
	plt.close(fig2)
	
	
if __name__ == '__main__':
	
	train()
