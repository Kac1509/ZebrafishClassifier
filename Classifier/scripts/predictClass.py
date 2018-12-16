import numpy as np
import shutil
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def predictClass(model,prediction_path, shapeY, shapeX, class_mode='binary'):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        prediction_path,
        target_size=(shapeY, shapeX),
        batch_size=1,
        shuffle = False,
        class_mode=class_mode)
    predictions = model.predict_generator(test_generator, verbose=1)
    return test_generator,predictions