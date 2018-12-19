import os
import glob
from createFolders import *
#Create Genotype classes
class Genotype:
    def __init__(self, name):
        self.name = name
        
def createGenotypes(Extracted_path,Partitioned_path):
    Genotypes = []
    num_classes = len(glob.glob(Extracted_path+'*'))
    for i in range(num_classes):
            #Retrieve directory for a given type
            Genotype_name = os.listdir(Extracted_path)[i]
            Genotypes.append(Genotype(Genotype_name))
    print(num_classes)
    createDirectories(num_classes,Extracted_path,Partitioned_path,Genotypes)
    return Genotypes
