import os
import sys
from PIL import Image
from random import shuffle
import glob
import shutil
import zipfile


def deleteFiles(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def createFolder(path):
    if os.path.exists(path):
        #Retry if you can't remove contents
        for retry in range(100):
            try:
                shutil.rmtree(path)
            except OSError as e:
                print ("Error: %s - %s." % (e.filename, e.strerror))
    os.makedirs(path)
    
def unzip_data(src_path, dst_path):
  local_zip = src_path
  zip_ref = zipfile.ZipFile(local_zip, 'r')
  zip_ref.extractall(dst_path)
  zip_ref.close()
    
    
def createDirectories(nbClasses,Paths,Genotypes):
    
    for i in range(nbClasses):
        #Retrieve directory for a given Genotype
        class_dir = Paths.extracted_path + Genotypes[i].name + '/'
        class_images = []

        #Retrieve all images link to a given Genotype
        for idx, filename in enumerate(os.listdir(class_dir)): 
            with Image.open(class_dir+filename) as im:
                im=Image.open(class_dir+filename)
                class_images.append(im)
            

        #Save images to Genotype class 
        Genotypes[i].images = class_images  

        #Create Train and Test Folders for each Genotype
        Genotypes[i].train_path = Paths.partitioned_path + 'Train/' + Genotypes[i].name  + '/'
        Genotypes[i].validation_path = Paths.partitioned_path + 'Validation/' + Genotypes[i].name  + '/'
        createFolder(Genotypes[i].train_path)
        createFolder(Genotypes[i].validation_path)

def saveFiles(Genotypes):
     for i in range(len(Genotypes)):
            
            #Save train and validation paritions for each Genotype
            for idx in range(len(Genotypes[i].trainSet)):
                save_fname0 = os.path.join(Genotypes[i].train_path, 'Train' + str(idx+1) + '.png')
                Genotypes[i].trainSet[idx].save(save_fname0)
            for idx in range(len(Genotypes[i].testSet)):
                save_fname0 = os.path.join(Genotypes[i].validation_path, 'Validation' + str(idx+1) + '.png')
                Genotypes[i].testSet[idx].save(save_fname0)