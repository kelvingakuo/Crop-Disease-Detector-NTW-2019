import cv2
import glob
import pickle
from pathlib import Path

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array


def present_tests(model, classes):
	""" Select images from test folders(s), display true and predicted classes
	Params:
		model - Loaded Keras model
		classes - Classes loaded from binarizer
	"""
	folders = glob.glob("../test_data/*") 
	true = [Path(folder).parts[2] for folder in folders]


	k = 0
	while (k < len(true)):
		trueLbl = true[k] 
		folder = folders[k]

		#imgs = glob.glob(folder+'/*.JPG')
		images = glob.glob(folder + '/*.jpg')
		
		#images = imgs1 + imgs

		print(len(images))
		for image in images:
			rd = cv2.imread(image)
			arr = cv2.resize(rd, (256, 256))
			arr = arr.astype('uint64')
			theImg = img_to_array(arr)
			theImg = theImg.reshape([1, 256, 256, 3])


			pred = model.predict_classes(theImg)
			prob = model.predict_proba(theImg)

			print(pred)


			print('True label: {}'.format(trueLbl))
			print('Predicted Label: {}'.format(classes[pred[0]]))
			print('\n')

		k = k + 1





if __name__ == '__main__':
	model = load_model('../trained_models/model.h5')

	with open('../trained_models/label_transform.pkl', 'rb') as biner:
		binarizer = pickle.load(biner)

	classes = binarizer.classes_


	present_tests(model, classes)