import os
from createFolders import *
import numpy as np

#Retrieve the size of the type with the fewest images 
def retrieveSmall(Genotypes,size):
    for i in range(len(Genotypes)):
        print(len(Genotypes[i].images))
        if(len(Genotypes[i].images)<size):
            size = len(Genotypes[i].images)
            idx = i
    return idx,size

#Shuffle images
#Fixed: Fixed size of training and test set for all types
#Otherwise: Variable size of training and test set for all types
def shuffleImages_split(Genotype,training_size, size,fixed):
    shuffle(Genotype.images)
    if fixed:
        training_split = int(np.round(training_size*size))
        end_split = size
    else:
        training_split = int(np.round(training_size*len(Genotype.images)))
        end_split = -1
    Genotype.trainSet = Genotype.images[:training_split]
    Genotype.testSet = Genotype.images[training_split:end_split]

#Create Training and Test set
def createTrain_Test(Genotypes,training_size,fixed = False):
    training_size = training_size
    idx,size = retrieveSmall(Genotypes,sys.maxsize)
    for i in range(len(Genotypes)):
        shuffleImages_split(Genotypes[i],training_size,size,fixed)

