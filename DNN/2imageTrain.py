'''


Train on XZ and YZ views

https://stackoverflow.com/questions/51200821/keras-layer-concatenation
https://www.programcreek.com/python/example/89660/keras.layers.concatenate
https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
https://www.kaggle.com/hireme/two-inputs-neural-network-using-keras

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
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation, Input, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, ReduceLROnPlateau
from keras import regularizers
from keras.optimizers import Adam


def getModel():
	'''
	https://www.kaggle.com/adityaecdrid/mnist-with-keras-for-beginners-99457/notebook
	
	'''
	
	# XZ branch
	XZ_in = Input( shape=(7,22,1), name='input1' )
	xz = Conv2D(32, kernel_size=(3, 3),kernel_initializer='he_normal')(XZ_in)
	xz = Activation('relu')(xz)
	xz = Conv2D(32, kernel_size=(3, 3),kernel_initializer='he_normal')(xz)
	xz = Activation('relu')(xz)
	xz = MaxPooling2D((2,2))(xz)
	xz = Dropout(0.20)(xz)
	xz = Conv2D(64, kernel_size=(3, 3),kernel_initializer='he_normal',padding='same')(xz)
	xz = Activation('relu')(xz)
	xz = Conv2D(64, kernel_size=(3, 3),kernel_initializer='he_normal',padding='same')(xz)
	xz = Activation('relu')(xz)
	xz = MaxPooling2D(pool_size=(1,2))(xz)
	xz = Dropout(0.25)(xz)
	xz = Conv2D(128, kernel_size=(3,3),kernel_initializer='he_normal',padding='same')(xz)
	xz = Activation('relu')(xz)
	xz = Dropout(0.25)(xz)
	xz = Flatten()(xz)
	
	# YZ branch
	YZ_in = Input( shape=(7,22,1), name='input2' )
	yz = Conv2D(32, kernel_size=(3, 3),kernel_initializer='he_normal')(YZ_in)
	yz = Activation('relu')(yz)
	yz = Conv2D(32, kernel_size=(3, 3),kernel_initializer='he_normal')(yz)
	yz = Activation('relu')(yz)
	yz = MaxPooling2D((2,2))(yz)
	yz = Dropout(0.20)(yz)
	yz = Conv2D(64, kernel_size=(3, 3),kernel_initializer='he_normal',padding='same')(yz)
	yz = Activation('relu')(yz)
	yz = Conv2D(64, kernel_size=(3, 3),kernel_initializer='he_normal',padding='same')(yz)
	yz = Activation('relu')(yz)
	yz = MaxPooling2D(pool_size=(1,2))(yz)
	yz = Dropout(0.25)(yz)
	yz = Conv2D(128, kernel_size=(3,3),kernel_initializer='he_normal',padding='same')(yz)
	yz = Activation('relu')(yz)
	yz = Dropout(0.25)(yz)
	yz = Flatten()(yz)
	
	x = concatenate( [xz,yz] )
	x = Dense(256)(x)
	x = Activation('relu')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.25)(x)
	out = Dense(1,activation='sigmoid')(x)
	
	model = Model( inputs=[XZ_in,YZ_in],outputs=out)
		
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
	
	return model
	
	

	
def getLinearModel(model):
	
	# XZ branch
	XZ_in = Input( shape=(7,22,1), name='input1' )
	xz = Conv2D(32, kernel_size=(3, 3),kernel_initializer='he_normal')(XZ_in)
	xz = Activation('relu')(xz)
	xz = Conv2D(32, kernel_size=(3, 3),kernel_initializer='he_normal')(xz)
	xz = Activation('relu')(xz)
	xz = MaxPooling2D((2,2))(xz)
	xz = Dropout(0.20)(xz)
	xz = Conv2D(64, kernel_size=(3, 3),kernel_initializer='he_normal',padding='same')(xz)
	xz = Activation('relu')(xz)
	xz = Conv2D(64, kernel_size=(3, 3),kernel_initializer='he_normal',padding='same')(xz)
	xz = Activation('relu')(xz)
	xz = MaxPooling2D(pool_size=(1,2))(xz)
	xz = Dropout(0.25)(xz)
	xz = Conv2D(128, kernel_size=(3,3),kernel_initializer='he_normal',padding='same')(xz)
	xz = Activation('relu')(xz)
	xz = Dropout(0.25)(xz)
	xz = Flatten()(xz)
	
	# YZ branch
	YZ_in = Input( shape=(7,22,1), name='input2' )
	yz = Conv2D(32, kernel_size=(3, 3),kernel_initializer='he_normal')(YZ_in)
	yz = Activation('relu')(yz)
	yz = Conv2D(32, kernel_size=(3, 3),kernel_initializer='he_normal')(yz)
	yz = Activation('relu')(yz)
	yz = MaxPooling2D((2,2))(yz)
	yz = Dropout(0.20)(yz)
	yz = Conv2D(64, kernel_size=(3, 3),kernel_initializer='he_normal',padding='same')(yz)
	yz = Activation('relu')(yz)
	yz = Conv2D(64, kernel_size=(3, 3),kernel_initializer='he_normal',padding='same')(yz)
	yz = Activation('relu')(yz)
	yz = MaxPooling2D(pool_size=(1,2))(yz)
	yz = Dropout(0.25)(yz)
	yz = Conv2D(128, kernel_size=(3,3),kernel_initializer='he_normal',padding='same')(yz)
	yz = Activation('relu')(yz)
	yz = Dropout(0.25)(yz)
	yz = Flatten()(yz)
	
	x = concatenate( [xz,yz] )
	x = Dense(256)(x)
	x = Activation('relu')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.25)(x)
	out = Dense(1)(x)
	
	model2 = Model( inputs=[XZ_in,YZ_in],outputs=out)
	model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
	
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
	
	X_train_XZ = X_train[:,[2*i for i in range(7)],:,:]
	X_train_YZ = X_train[:,[2*i+1 for i in range(7)],:,:]
	X_test_XZ = X_test[:,[2*i for i in range(7)],:,:]
	X_test_YZ = X_test[:,[2*i+1 for i in range(7)],:,:]
	
	model = getModel()
	rdlronplt = ReduceLROnPlateau(monitor='loss',patience=3,verbose=1,factor=0.5,min_lr=0.0001)
	modelcheckpoint = ModelCheckpoint('out/weights.{epoch:02d}-{val_loss:.4f}.hdf5',save_weights_only=True)
	
	history = model.fit( [X_train_XZ,X_train_YZ],y_train ,batch_size=100,epochs=140,verbose=2,callbacks=[rdlronplt,modelcheckpoint],validation_data=([X_test_XZ,X_test_YZ],y_test))
	
	predictions = model.predict([X_test_XZ,X_test_YZ])
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
	
	
	model2 = getLinearModel(model)
	model2.save('out/model.h5')
	predictions = model2.predict([X_test_XZ,X_test_YZ])
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
	binList = np.linspace(-20,20,100)
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
