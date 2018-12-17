from PIL import Image
from random import shuffle
import glob
import os
import shutil


def separete_train_test_data(src_path, training_size = 20):
  # Images input
  class0_list = []
  for filename in glob.glob(src_path+'/Data/her1her7s/*.png'): #assuming png
      im=Image.open(filename)
      class0_list.append(im)

  class1_list = []
  for filename in glob.glob(src_path+'/Data/fsstbx6s/*.png'): #assuming png
      im=Image.open(filename)
      class1_list.append(im)

  # Images shuffle
  shuffle(class0_list)
  class0_train_set = class0_list[:training_size]
  class0_test_set = class0_list[training_size:]
  shuffle(class1_list)
  class1_train_set = class1_list[:training_size]
  class1_test_set = class1_list[training_size:]

  # Images storing
  out_path_class0_train = src_path+'/Data/train/her1her7s'
  out_path_class1_train = src_path+'/Data/train/fsstbx6s'
  out_path_class0_test = src_path+'/Data/test/her1her7s'
  out_path_class1_test = src_path+'/Data/test/fsstbx6s'

  shutil.rmtree(out_path_class0_train)
  os.makedirs(out_path_class0_train)
  shutil.rmtree(out_path_class1_train)
  os.makedirs(out_path_class1_train)
  shutil.rmtree(out_path_class0_test)
  os.makedirs(out_path_class0_test)
  shutil.rmtree(out_path_class1_test)
  os.makedirs(out_path_class1_test)

  # construct output filename, basename to remove input directory
  for i in range(len(class1_train_set)):
      save_fname0 = os.path.join(out_path_class0_train, 'Train' + str(i+1) + '.png')
      class0_train_set[i].save(save_fname0)
      save_fname1 = os.path.join(out_path_class1_train, 'Train' + str(i+1) + '.png')
      class1_train_set[i].save(save_fname1)


  # construct output filename, basename to remove input directory
  for i in range(len(class1_test_set)):
      save_fname0 = os.path.join(out_path_class0_test, 'Test' + str(i+1) + '.png')
      class0_test_set[i].save(save_fname0)
      save_fname1 = os.path.join(out_path_class1_test, 'Test' + str(i+1) + '.png')
      class1_test_set[i].save(save_fname1)