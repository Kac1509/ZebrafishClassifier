from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
from PIL import Image
import glob
from random import shuffle
import shutil
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	# return cv2.cvtColor(cv2.resize(np.array(image), size), cv2.COLOR_BGR2GRAY).flatten()
  return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(32, 32, 32)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 256, 0, 256, 0, 256])

	cv2.normalize(hist, hist)

	# return the flattened histogram as the feature vector
	return hist.flatten()
	
	
def read_images(path):
  # Images input
  arr = []
  for filename in glob.glob(path): #assuming png
      im=Image.open(filename)
      arr.append(im)
  # print len(arr)
  return arr
  
def images_to_rawpixels(class0, class1, class2, shape_x, shape_y):
    
    # initialize the raw pixel intensities matrix, the features matrix,
    # and labels list
    rawImages = []
    labels = []

    images = class0 + class1 + class2
    # print len(images)

    for i in range(len(images)):
      # load the image and extract the class label 
      image = np.array(images[i])
      label = 'HH' if i<len(class0) else 'Fss'

      # extract raw pixel intensity "features"
      # in the image
      pixels = image_to_feature_vector(image, (shape_x, shape_y))

      # update the raw images and labels matricies,
      # respectively
      rawImages.append(pixels)
      labels.append(label)

    # show some information on the memory consumed by the raw images
    # matrix and features matrix
    rawImages = np.array(rawImages)
    labels = np.array(labels)
    print("[INFO] pixels matrix: {:.2f}MB".format(
      rawImages.nbytes / (1024 * 1000.0)))
    return rawImages, labels
	
def images_to_hystogram_features(class0, class1, class2):
    
    # initialize the features matrix,
    # and labels list
    features = []
    labels = []

    images = class0 + class1 + class2
    # print len(images)

    for i in range(len(images)):
      # load the image and extract the class label 
      image = np.array(images[i])
      label = 'HH' if i<len(class0) else 'Fss'

      # extract color histogram to characterize the color distribution of the pixels
      # in the image
      hist = extract_color_histogram(image)

      # update the features, and labels matricies,
      # respectively
      features.append(hist)
      labels.append(label)

    # show some information on the memory consumed by the raw images
    # matrix and features matrix
    features = np.array(features)
    labels = np.array(labels)
    print("[INFO] features matrix: {:.2f}MB".format(
      features.nbytes / (1024 * 1000.0)))
    return features, labels
	

def train_knn(n_neighbours, inputX, labels):
  # train and evaluate a k-NN classifer on the raw pixel intensities
  # print("[INFO] evaluating raw pixel accuracy...")
  model = KNeighborsClassifier(n_neighbors=n_neighbours,
    n_jobs=-1)
  model.fit(inputX, labels)
  return model
  
def evaluate_knn(model, inputX, labels):
  
  # evaluate a k-NN classifer
  acc = model.score(inputX, labels)
  print("[INFO] model accuracy: {:.2f}%".format(acc * 100))
  certainty = model.predict_proba(inputX)
  print("[INFO] model certainty: [{:.2f}%:{:.2f}%]"
        .format(np.average(certainty[np.nonzero(certainty[:,0]>0.5), 0]),np.average(certainty[np.nonzero(certainty[:,1]>0.5), 1])))
  return acc, certainty
  



def separete_train_test_data(src_path, training_size = 20):
  # Images input
  class0_list = []
  for filename in glob.glob(src_path+'/her1her7s/*.png'): #assuming png
      im=Image.open(filename)
      class0_list.append(im)

  class1_list = []
  for filename in glob.glob(src_path+'/fsstbx6s/*.png'): #assuming png
      im=Image.open(filename)
      class1_list.append(im)
	  
  class2_list = []
  for filename in glob.glob(src_path+'/WTs/*.png'): #assuming png
      im=Image.open(filename)
      class1_list.append(im)

  # Images shuffle
  shuffle(class0_list)
  class0_train_set = class0_list[:training_size]
  class0_test_set = class0_list[training_size:]
  shuffle(class1_list)
  class1_train_set = class1_list[:training_size]
  class1_test_set = class1_list[training_size:]
  shuffle(class2_list)
  class2_train_set = class1_list[:training_size]
  class2_test_set = class1_list[training_size:]

  # Images storing
  out_path_class0_train = src_path+'/Train/her1her7s'
  out_path_class1_train = src_path+'/Train/fsstbx6s'
  out_path_class2_train = src_path+'/Train/WTs'
  out_path_class0_test = src_path+'/Test/her1her7s'
  out_path_class1_test = src_path+'/Test/fsstbx6s'
  out_path_class2_test = src_path+'/Test/WTs'

  if os.path.isdir(out_path_class0_train):
    shutil.rmtree(out_path_class0_train)
  os.makedirs(out_path_class0_train)
  if os.path.isdir(out_path_class1_train):
    shutil.rmtree(out_path_class1_train)
  os.makedirs(out_path_class1_train)
  if os.path.isdir(out_path_class2_train):
    shutil.rmtree(out_path_class2_train)
  os.makedirs(out_path_class2_train)
  if os.path.isdir(out_path_class0_test):
    shutil.rmtree(out_path_class0_test)
  os.makedirs(out_path_class0_test)
  if os.path.isdir(out_path_class1_test):
    shutil.rmtree(out_path_class1_test)
  os.makedirs(out_path_class1_test)
  if os.path.isdir(out_path_class2_test):
    shutil.rmtree(out_path_class2_test)
  os.makedirs(out_path_class2_test)

  # construct output filename, basename to remove input directory
  for i in range(len(class1_train_set)):
      save_fname0 = os.path.join(out_path_class0_train, 'Train' + str(i+1) + '.png')
      class0_train_set[i].save(save_fname0)
      save_fname1 = os.path.join(out_path_class1_train, 'Train' + str(i+1) + '.png')
      class1_train_set[i].save(save_fname1)
      save_fname2 = os.path.join(out_path_class2_train, 'Train' + str(i+1) + '.png')
      class2_train_set[i].save(save_fname1)


  # construct output filename, basename to remove input directory
  for i in range(len(class1_test_set)):
      save_fname0 = os.path.join(out_path_class0_test, 'Test' + str(i+1) + '.png')
      class0_test_set[i].save(save_fname0)
      save_fname1 = os.path.join(out_path_class1_test, 'Test' + str(i+1) + '.png')
      class1_test_set[i].save(save_fname1)
      save_fname2 = os.path.join(out_path_class2_test, 'Test' + str(i+1) + '.png')
      class2_test_set[i].save(save_fname2)