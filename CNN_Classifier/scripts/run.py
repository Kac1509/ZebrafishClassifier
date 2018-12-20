import sys
import os

#Set if running locally or in Google Col}aboratory
Colab = False
if Colab:
    Script_path = '/content/gdrive/My Drive/Colab Notebooks/Classifier/scripts'
    Base_path = '/content/gdrive/My Drive/Colab Notebooks/Classifier/'
    from google.colab import drive
    drive.mount('/content/gdrive')
else:
    Script_path = os.getcwd() + '/Classifier/scripts'
    Base_path = "C:/Users/Kaleem/EPFL/Fall 2018/Machine Learning/Project_2/Git/ZebrafishClassifier/Classifier/"

#Add script folder to system path
sys.path.insert(0, Script_path)
print(sys.path)


from CNN_Model import *
from DataVisualization import *
from Genotype import *
from HyperparameterTuning import *
from Predictions import *
from Setup import *

Paths = setupEnvironment(Base_path,'DataStraightened.zip')

Genotypes = createGenotypes(Paths)

Tuning = False
if Tuning:

    #These are the hyperparameters tested
    #learning_rates = np.logspace(-7, -2, 7)
    #hidden_nodes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    #input_sizes = [[75,750],[75,375], [75,150], [75,75], [50,50]]
    dropout = np.linspace(0.1, 0.5, 5)
    Plot_Xlabel = 'Dropout Rate'


    Parameters = setParameters(Paths, shapeY = 50, shapeX = 50,
                                    dropout_rate = 0,
                                    LR = 0.0001,
                                    num_nodes = 256,
                                    VGG16 = True,
                                    Dropout = False)
    Hyperparameters,Losses = Hyperparameter_tuning(Paths,
                                                   Genotypes,
                                                   Parameters,
                                                   Hyperparameter = dropout,
                                                   epochs = 3,
                                                   k_fold = 2)

    cross_validation_visualization(Hyperparameters,Losses[0], Losses[1],Losses[2], Losses[3],Plot_Xlabel)

else:
    #Partition data into training and test set
    createTrain_Test(Genotypes,training_size = 0.7, fixed = True)

    Parameters = setParameters(Paths,
                               shapeY = 100, shapeX = 100,
                               dropout_rate = 0,
                               LR = 0.0001,
                               num_nodes = 256,
                               VGG16 = True,
                               Dropout = False)

    Hist,Mdl = runModel(Paths,Genotypes,Parameters,epochs=20)

    plot_loss_acc(Hist)


    #Format predictions to 4 decimal points
    float_formatter = lambda x: "%.4f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter})

    #Predict Inages in prediction path and save predictions to a csv file
    predictionFiles, predictions = predictClass(Mdl,Paths, Parameters, class_mode='categorical')
    savePredictions(Paths,predictions,predictionFiles,Parameters)

