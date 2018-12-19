import sys
import os
from createFolders import *
from unzip_data import *

def setupEnvironment(Colab = False):
    #Set if running locally or in Google Col}aboratory
    if Colab:
        Script_path = '/content/gdrive/My Drive/Colab Notebooks/Classifier/scripts'
        Base_path = '/content/gdrive/My Drive/Colab Notebooks/Classifier/'
        from google.colab import drive
        drive.mount('/content/gdrive')
    else:
        Script_path = os.getcwd() + '/Classifier/scripts'
        Base_path = 'Classifier/'

    #Add script folder to system path
    sys.path.insert(0, Script_path)
    print(sys.path)
    

    
def setupEnv(Base_path,Data_file):
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

    