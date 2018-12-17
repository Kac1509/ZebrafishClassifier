import sys
sys.path.insert(0, 'scripts')
#print sys.path

import os
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)

from unzip_data import *
from separete_train_test_data import *
from KNN_helpers import *

# builds and evaluates two knn models
def run_knn(n_neighbours = 3):

	zip_path = 'Data.zip'
	folder_path = 'tmp'

	shapeY = 150
	shapeX = 750
	color_channels = 3


	# Unziping(if necessary) and separating data into Train and Validation sets
	if os.path.isdir(folder_path+'/Data') == False:
	  unzip_data(zip_path, folder_path)
	  
	separete_train_test_data(folder_path, training_size = 20)

	# Readin Test and Train images
	print('TRAIN')
	class0_train = read_images(folder_path+'/Data/train/her1her7s/*.png')
	class1_train = read_images(folder_path+'/Data/train/fsstbx6s/*.png')
	rawImages, _ = images_to_rawpixels(class0_train, class1_train, shapeX, shapeY)
	features, labels = images_to_hystogram_features(class0_train, class1_train)


	print('TEST')
	class0_test = read_images(folder_path+'/Data/test/her1her7s/*.png')
	class1_test = read_images(folder_path+'/Data/test/fsstbx6s/*.png')
	rawImagesTest, _ = images_to_rawpixels(class0_test, class1_test, shapeX, shapeY)
	featuresTest, labelsTest = images_to_hystogram_features(class0_test, class1_test)


	# Training and evaluation of the model
	modelR = train_knn(n_neighbours, rawImages, labels)
	modelH = train_knn(n_neighbours, features, labels)
	print('\nRaw pixels')
	evaluate_knn(modelR, rawImagesTest, labelsTest)
	print('\nHystogram')
	evaluate_knn(modelH, featuresTest, labelsTest)
	
	
	
# builds and evaluates knn two models multiple(k) times and calculates mean accuaracy of models  	
def run_knn_cross(n_neighbours = 3, k = 4):

	zip_path = 'Data.zip'
	folder_path = 'tmp'

	shapeY = 150
	shapeX = 750
	color_channels = 3


	# Unziping(if necessary) and separating data into Train and Validation sets
	if os.path.isdir(folder_path+'/Data') == False:
	  unzip_data(zip_path, folder_path)
	
	accRarr = []
	accHarr = []
	for	i in range(k):
		separete_train_test_data(folder_path, training_size = 20)

		# Readin Test and Train images
		print('\nTRAIN')
		class0_train = read_images(folder_path+'/Data/train/her1her7s/*.png')
		class1_train = read_images(folder_path+'/Data/train/fsstbx6s/*.png')
		rawImages, _ = images_to_rawpixels(class0_train, class1_train, shapeX, shapeY)
		features, labels = images_to_hystogram_features(class0_train, class1_train)


		print('TEST')
		class0_test = read_images(folder_path+'/Data/test/her1her7s/*.png')
		class1_test = read_images(folder_path+'/Data/test/fsstbx6s/*.png')
		rawImagesTest, _ = images_to_rawpixels(class0_test, class1_test, shapeX, shapeY)
		featuresTest, labelsTest = images_to_hystogram_features(class0_test, class1_test)


		# Training and evaluation of the model
		modelR = train_knn(n_neighbours, rawImages, labels)
		modelH = train_knn(n_neighbours, features, labels)
		print('\nRaw pixels')
		accR, cerR = evaluate_knn(modelR, rawImagesTest, labelsTest)
		accRarr.append(accR)
		print('\nHystogram')
		accH, cerH = evaluate_knn(modelH, featuresTest, labelsTest)
		accHarr.append(accH)
		
	print('\nRaw pixels mean accuracy: ' + str(np.mean(accRarr)))
	print('\nHystogram mean accuracy: ' + str(np.mean(accHarr)))
		
		