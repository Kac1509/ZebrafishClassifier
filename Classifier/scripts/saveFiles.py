import os
from createFolders import *
#Save Training and test images in their respective folders
def saveFiles(Genotypes):
     for i in range(len(Genotypes)):
            #deleteFiles(Genotypes[i].train_path)
            #deleteFiles(Genotypes[i].validation_path)
            for idx in range(len(Genotypes[i].trainSet)):
                save_fname0 = os.path.join(Genotypes[i].train_path, 'Train' + str(idx+1) + '.png')
                Genotypes[i].trainSet[idx].save(save_fname0)
            for idx in range(len(Genotypes[i].testSet)):
                save_fname0 = os.path.join(Genotypes[i].validation_path, 'Validation' + str(idx+1) + '.png')
                Genotypes[i].testSet[idx].save(save_fname0)