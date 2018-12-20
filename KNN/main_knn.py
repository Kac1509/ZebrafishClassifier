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
from KNN_helpers import *
from Helpers import *
from Setup import *
Script_path = os.getcwd() + '/scripts'
Base_path = ''

# builds and evaluates two knn models
def run_knn(n_neighbours = 3):

	shapeY = 150
	shapeX = 750
	color_channels = 3


	# Unziping(if necessary) and separating data into Train and Validation sets
	Paths = setupEnvironment(Base_path,'DataStraightened.zip')
	separete_train_test_data('ExtractedData', training_size = 20)

	# Readin Test and Train images
	print('TRAIN')
	class0_train = read_images('ExtractedData/Train/her1her7s/*.png')
	class1_train = read_images('ExtractedData/Train/fsstbx6s/*.png')
	class2_train = read_images('ExtractedData/Train/WTs/*.png')
	rawImages, _ = images_to_rawpixels(class0_train, class1_train, class2_train, shapeX, shapeY)
	features, labels = images_to_hystogram_features(class0_train, class1_train, class2_train)


	print('TEST')
	class0_test = read_images('ExtractedData/Test/her1her7s/*.png')
	class1_test = read_images('ExtractedData/Test/fsstbx6s/*.png')
	class2_test = read_images('ExtractedData/Test/WTs/*.png')
	rawImagesTest, _ = images_to_rawpixels(class0_test, class1_test, class2_test, shapeX, shapeY)
	featuresTest, labelsTest = images_to_hystogram_features(class0_test, class1_test, class2_test)


	# Training and evaluation of the model
	modelR = train_knn(n_neighbours, rawImages, labels)
	modelH = train_knn(n_neighbours, features, labels)
	print('\nRaw pixels')
	evaluate_knn(modelR, rawImagesTest, labelsTest)
	print('\nHystogram')
	evaluate_knn(modelH, featuresTest, labelsTest)
	
	
	
# builds and evaluates knn two models multiple(k) times and calculates mean accuaracy of models  	
def run_knn_cross(n_neighbours = 3, k = 4):

	shapeY = 150
	shapeX = 750
	color_channels = 3


	# Unziping(if necessary) and separating data into Train and Validation sets
	Paths = setupEnvironment(Base_path,'DataStraightened.zip')
	
	accRarr = []
	accHarr = []
	for	i in range(k):
		separete_train_test_data('ExtractedData', training_size = 20)

		# Readin Test and Train images
		print('TRAIN')
		class0_train = read_images('ExtractedData/Train/her1her7s/*.png')
		class1_train = read_images('ExtractedData/Train/fsstbx6s/*.png')
		class2_train = read_images('ExtractedData/Train/WTs/*.png')
		rawImages, _ = images_to_rawpixels(class0_train, class1_train, class2_train, shapeX, shapeY)
		features, labels = images_to_hystogram_features(class0_train, class1_train, class2_train)


		print('TEST')
		class0_test = read_images('ExtractedData/Test/her1her7s/*.png')
		class1_test = read_images('ExtractedData/Test/fsstbx6s/*.png')
		class2_test = read_images('ExtractedData/Test/WTs/*.png')
		rawImagesTest, _ = images_to_rawpixels(class0_test, class1_test, class2_test, shapeX, shapeY)
		featuresTest, labelsTest = images_to_hystogram_features(class0_test, class1_test, class2_test)



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
		
		