import cv2
import glob
import pickle

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array



with open('../trained_models/label_transform.pkl', 'rb') as biner:
	binarizer = pickle.load(biner)

classes = binarizer.classes_
for cls in classes:
	print(cls)
	vec = binarizer.transform([cls])
	lbl = np.argmax(vec, axis = 1)
	lb = lbl[0]
	with open('../trained_models/predicted_diseases.txt', 'a') as op:
		op.write(cls)
		op.write('\n')
		op.write(str(lb))
		op.write('\n\n')

'''
model = load_model('../trained_models/model.h5')
print(model.summary())

testImgs = glob.glob('../test_data'+'/*.jpg')
JPGs = glob.glob('../test_data'+'/*.JPG')
allTestImgs = testImgs + JPGs

for img in allTestImgs:
	rd = cv2.imread(img)
	arr = cv2.resize(rd, (256, 256))
	arr = arr.astype('uint64')
	theImg = img_to_array(arr)
	theImg = theImg.reshape([1, 256, 256, 3])


	pred = model.predict_classes(theImg)
	prob = model.predict_proba(theImg)

	print(img)
	print(pred)
	print(prob)

	print(classes[pred[0]])
	print('\n\n\n\n')
'''