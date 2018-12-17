import os
import sys
from PIL import Image
from random import shuffle
import glob
import shutil



def deleteFiles(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def createFolder(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))
    os.makedirs(path)
    
def createDirectories(nbClasses,data_dir,base_dir,Genotypes):
    
    for i in range(nbClasses):
        #Retrieve directory for a given type
        class_dir = data_dir + Genotypes[i].name + '/'
        class_images = []

        #Retrieve all images link to a given type
        for idx, filename in enumerate(os.listdir(class_dir)): 
            with Image.open(class_dir+filename) as im:
                im=Image.open(class_dir+filename)
                class_images.append(im)
            

        #Create Zebra Object and save images to type 
        Genotypes[i].images = class_images  

        #Create Train and Test Folders for each type
        Genotypes[i].train_path = base_dir + 'Train/' + Genotypes[i].name  + '/'
        Genotypes[i].validation_path = base_dir + 'Validation/' + Genotypes[i].name  + '/'
        createFolder(Genotypes[i].train_path)
        createFolder(Genotypes[i].validation_path)
