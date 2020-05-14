import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import models
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import pandas as pd
import os, re, glob
import pickle as pkl

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_img_tensor(img_path):
	import cv2
	img = cv2.imread(img_path, 0)
	img_tensor = np.expand_dims(np.expand_dims(cv2.resize(img, (48, 48)), -1), 0)
	return img_tensor

def padding_and_average(mp, max_uttr):
	for dia in mp:
		for uttr in range(max_uttr):
			if not mp[dia][uttr]:
				mp[dia][uttr] = np.zeros(activation_shape)
			else:
				mp[dia][uttr] = np.average(mp[dia][uttr], axis=0)
	return mp

def get_activation(img_tensor, classifier):
	layer_outputs = [layer.output for layer in classifier.layers[:]] # Extracts the outputs of the top 12 layers
	activation_model = models.Model(inputs=classifier.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
	activations = activation_model.predict(img_tensor) # Returns a list of five Numpy arrays: one array per layer activation
	return activations[16]

# Define data generators
train_dir = 'data/train'
val_dir = 'data/dev'
test_dir = 'data/test'
dirs = [train_dir, val_dir, test_dir]

classifier = load_model('model.h5')

for dir_ in dirs:
	print(dir_)
	mp = {}
	max_uttr = 33 #this is the max utter index
	activation_shape = 0
	flag = False
	
	print('Getting activation maps for all images...')
	for f in glob.glob(os.path.join(dir_, '**/*')):
		fname = os.path.basename(f)
		num = re.split(r'\D+', fname, flags=re.IGNORECASE)
		dia  = num[1]
		uttr = int(num[2])
		img_tensor = get_img_tensor(f)
		if dia not in mp:
			mp[dia] = np.array([None]*max_uttr)
			for i in range(max_uttr):
				mp[dia][i] = []
		mp[dia][uttr].append(get_activation(img_tensor, classifier))
		if not flag:
			activation_shape = mp[dia][uttr][0].shape
			flag = True
	
	print("Converting to the desired format...")	
	mp = padding_and_average(mp, max_uttr)
	reshape_mp = {}
	for key, value in mp.items():
		reshape_mp[key] = np.array([None]*max_uttr)
		for i in range(len(value)):
			reshape_mp[key][i] = value[i].reshape(-1)
	with open('{}_activation_emb.pkl'.format(os.path.basename(dir_)), 'wb') as f:
		pkl.dump(mp, f)
