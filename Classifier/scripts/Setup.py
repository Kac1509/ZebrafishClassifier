import sys
import os
from Helpers import *

   
def setupEnvironment(Base_path,Data_file):
    #Set Paths
    Zip_path = Base_path + Data_file
    Extracted_path = Base_path + 'ExtractedData/'
    Partitioned_path = Base_path + 'PartitionedData/'
    Prediction_path = Base_path + 'Predictions/'
    Validation_prediction_path = Base_path + 'Validation_Predictions/'
    
    #Clear Folders and create Prediction folder
    deleteFiles(Extracted_path)
    deleteFiles(Partitioned_path)
    createFolder(Prediction_path)
    
    #Extract Data
    unzip_data(Zip_path, Extracted_path)
    
    return Extracted_path,Partitioned_path,Prediction_path,Validation_prediction_path

    