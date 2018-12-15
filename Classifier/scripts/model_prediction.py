import numpy as np
import shutil
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def model_prediction(model, Genotypes, prediction_path,shapeY, shapeX, class_mode='binary'):
  
  
  if os.path.isdir(prediction_path):
      shutil.rmtree(prediction_path)
    
  # Note that the test data should not be augmented!
  test_datagen = ImageDataGenerator(rescale=1./255)
  for i in range(len(Genotypes)):
    Genotypes[i].prediction_path = prediction_path+Genotypes[i].name
    shutil.copytree(Genotypes[i].validation_path, Genotypes[i].prediction_path+'/test')
     
    Genotypes[i].test_generator = test_datagen.flow_from_directory(
          Genotypes[i].prediction_path,
          target_size=(shapeY, shapeX),
          batch_size=1,
          class_mode=class_mode)
    Genotypes[i].prediction = model.predict_generator(Genotypes[i].test_generator, verbose=1)
    print(Genotypes[i].name)
    print(Genotypes[i].prediction)
    