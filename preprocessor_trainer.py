# A simpler version was used for training on Kaggle: https://www.kaggle.com/kelvingakuo/one-yuuge-script/

import cv2
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import os
import pickle
import random
import re
import sys
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from keras.models import load_model
from keras.models import Model,Sequential
from keras.layers import Flatten,Dense,Dropout,Activation,Input,Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.losses import categorical_crossentropy
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator

form = logging.Formatter("%(asctime)s : %(levelname)-5.5s : %(message)s")
logger = logging.getLogger()


consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(form)
logger.addHandler(consoleHandler)

logger.setLevel(logging.DEBUG)

def convert_img_to_array(img,dim):
	try:
		image= load_img(img, target_size=(dim,dim))
		return img_to_array(image)
	except Exception as e:
		print(f"Error : {e}")	
		return None	
		

def preprocessing(dim):
	folders = glob.glob("../input/repository/spMohanty-PlantVillage-Dataset-442d23a/raw/color/*") #Kaggle dir
	parts = [Path(folder).parts[6] for folder in folders]
	
	needed = ['Tomato', 'Potato', 'Pepper']
	wanted = [part for part in parts if any(x in part for x in needed)]
	
	classes = len(wanted)
	
	train_data = dict()
	train_data['Image'] = []
	train_data['label'] = []

	k = 0
	while (k < len(wanted)):
		lbl = wanted[k] 
		logger.info('Opening folder.... {}'.format(lbl))
		folder = folders[k] 

		imgs = glob.glob(folder+'/*.JPG')
		imgs1 = glob.glob(folder + '/*.jpg')
		
		images = imgs1 + imgs
		
		
		random.shuffle(images)

		for image in images[:300]:

			arr = convert_img_to_array(image,dim)

			arr = np.array(arr, dtype= np.float16)/ 255.0

			train_data['Image'].append(arr)
			train_data['label'].append(lbl)

		logger.info('These many folders have been loaded: {}.... '.format(k+1))

		k += 1

	logger.info('Finished creating arrays. Writing to DF...')
	np.set_printoptions(threshold = np.inf)
	df = pd.DataFrame.from_dict(train_data, orient = 'columns')
	logger.info('Finished writing to DF.')					

	return df, classes


def alexnet(classes):
	model = Sequential()
	#convolution layer 1
	model.add(Conv2D(96,kernel_size=(11,11),strides=(4,4),padding='valid', data_format = 'channels_last', input_shape = (227, 227, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	model.add(BatchNormalization())

	#Convolution layer 2
	model.add(Conv2D(filters=256,kernel_size=(11,11),strides=(1,1),padding='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	model.add(BatchNormalization())

	#Convolution layer 3
	model.add(Conv2D(filters=384,kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	#Convolution layer 4
	model.add(Conv2D(filters=384,kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	#convolution layer 5
	model.add(Conv2D(filters=256,kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	model.add(BatchNormalization())

	model.add(Flatten())

	#Fully connected layer 1
	model.add(Dense(4096, input_shape = (227, 227, 3), activation='relu'))
	model.add(Dropout(0.4))
	model.add(BatchNormalization())

	#Fully  connected layer 2
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.4))
	model.add(BatchNormalization())

	#Fully connected layer 3
	model.add(Dense(classes,activation='relu')) # Output size of 24 classes
	model.add(Dropout(0.4))

	#output layer
	model.add(Activation('softmax'))
	
	optim = Adam(lr = 1e-3, decay = 1e-3 / 50)
	model.compile(loss = 'binary_crossentropy', optimizer = optim, metrics = ['accuracy'])
	
	return model


def some_model(classes):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), padding="same",  data_format = 'channels_last', input_shape = (256, 256, 3)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = -1))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding="same",activation="relu"))
    model.add(BatchNormalization(axis= -1))
    model.add(Conv2D(64, (3, 3), padding="same",activation="relu"))
    model.add(BatchNormalization(axis= -1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding="same",activation="relu"))
    model.add(BatchNormalization(axis= -1))
    model.add(Conv2D(128, (3, 3), padding="same",activation="relu"))
    model.add(BatchNormalization(axis= -1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024,activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(classes,activation="softmax"))
    
    optim = Adam(lr = 1e-3, decay = 1e-3 / 50)
    model.compile(loss = 'binary_crossentropy' ,optimizer= optim, metrics=['accuracy'])
    
    return model


def train(X, Y, classes, whichOne):
	XTrain, XTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=42)
	
	aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")

	if(whichOne == 'alexnet'):
		if(os.path.exists('agrix_alexnet.h5')):
			logger.info('Reading alexnet from file. Time to improve!!')
			model = load_model('agrix_alexnet.h5')
		else:
			logger.info('Instantiating model for the first time')
			model = alexnet(classes)

	else:
		if(os.path.exists('some_agrix_nn.h5')):
			logger.info('Reading some agrix nn from file. Time to improve!!')
			model = load_model('some_agrix_nn.h5')
		else:
			logger.info('Instantiating model for the first time')
			model = some_model(classes)


	model.fit_generator(aug.flow(XTrain, yTrain, batch_size = 32), epochs = 100, validation_data = (XTest, yTest), steps_per_epoch = len(XTrain) // 32, verbose = 1)
	scores = model.evaluate(XTest, yTest, batch_size = 32, verbose = 1)	
	logger.info("VALIDATION SCORE: {}: {}%".format(model.metrics_names[1], scores[1] * 100))

	if(whichOne == 'alexnet'):			
		model.save('agrix_alexnet.h5')
		logger.info('Saved alexnet to file.')
	else:
		model.save('agrix_alexnet.h5')
		logger.info('Saved some nn to file.')

	
	


if(name == "__main__")
	nn = sys.argv[1]

	if(nn == 'alexnet'):
		dim = 227
	else:
		dim = 256


	df, classes = preprocessing(dim) #Read all data

	X = np.array(df['Image'].tolist()) # Generate array of arrays for X, and array of vectors for y
	logger.info('Loaded X')

	label_binarizer = LabelBinarizer()
	Y = label_binarizer.fit_transform(df['label'])
	logger.info('Loaded Y')

	pickle.dump(label_binarizer, open('label_transform.pkl', 'wb'))
	logger.info('Wrote labels to file')

	logger.info('X.shape: {}'.format(X.shape))
	logger.info('Y.shape: {}'.format(Y.shape))
	logger.info('Starting training...')

	train(X, Y, classes, nn)