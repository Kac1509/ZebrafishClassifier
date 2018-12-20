import sys
import os
from Helpers import *

   
class Paths:
    def __init__(self,base_path):
        self.base_path = base_path
        
def setupEnvironment(Base_path,Data_file):
    #Set Paths
    Path = Paths(Base_path)
    Path.zip_path = Base_path + Data_file
    Path.extracted_path = Base_path + 'ExtractedData/'
    Path.partitioned_path = Base_path + 'PartitionedData/'
    Path.prediction_path = Base_path + 'Predictions/'
    Path.validation_prediction_path = Base_path + 'Validation_Predictions/'
    
    #Clear Folders and create Prediction folder
    deleteFiles(Path.extracted_path)
    deleteFiles(Path.partitioned_path)
    #createFolder(Path.prediction_path)
    
    #Extract Data
    unzip_data(Path.zip_path, Path.extracted_path)
    
    return Path

    