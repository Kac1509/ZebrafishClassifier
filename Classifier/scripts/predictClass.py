import numpy as np
import shutil
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import csv
import datetime
from tensorflow.keras.backend import eval


def predictClass(model,prediction_path, Hyperparameters, class_mode='binary'):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        prediction_path,
        target_size=(Hyperparameters.shapeY, Hyperparameters.shapeX),
        batch_size=1,
        shuffle = False,
        class_mode=class_mode)
    predictions = model.predict_generator(test_generator, verbose=1)
    return test_generator,predictions


def savePredictions(Base_path,predictions,predictionFiles,Hyperparameters):
    ModelType = 'Prediction_Model_' 
    time = f"{datetime.datetime.now():%Y-%m-%d_%H%M}"
    csvName = ModelType + time + '.csv'
    PredictionCSV = Base_path + csvName
    with open(PredictionCSV, 'w') as csvfile:
        fieldnames = ['Filename', 'WT', 'FSS', 'HH']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames,lineterminator = '\n')
        writer.writeheader()
        for i in range(len(predictions)):
            writer.writerow({'Filename':str(predictionFiles.filenames[i]),'WT':float(format(predictions[i][0], '.4f')),'FSS':float(format(predictions[i][1], '.4f')),'HH':float(format(predictions[i][2], '.4f'))})
        Model_parameters = "Learning Rate:", Hyperparameters.LR, ", Hidden_Units:", Hyperparameters.num_nodes, "Input_size:", Hyperparameters.shapeY, Hyperparameters.shapeX, "Dropout:", Hyperparameters.dropout_rate
        writer.writerow({'Filename':"Model",'WT':str(Model_parameters)})
