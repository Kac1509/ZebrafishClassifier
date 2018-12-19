from Helpers import *
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pre_trained_models import *

import os
import time

class Hyperparameters:
    def __init__(self,Tuning):
        self.Tuning = Tuning

def setParameters(Base_path, shapeY = 50, shapeX = 50, dropout_rate = 0, LR = 0.0001, num_nodes = 256, VGG16 = True):
    if VGG16:
        #VGG16 Model
        local_weights_file_VGG16 = Base_path + 'weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        pre_trained_model, last_layer_output = load_pre_trained_VGG16(local_weights_file_VGG16, shapeY, shapeX, color_channels = 3)
    else:
        # Inception Model
        local_weights_file_Inception = Base_path + 'weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        pre_trained_model, last_layer_output = load_pre_trained_Inception(local_weights_file_Inception, shapeY, shapeX, color_channels = 3)
    
    Hyperparm = Hyperparameters(True)
    Hyperparm.dropout_rate = dropout_rate
    Hyperparm.LR = LR
    Hyperparm.num_nodes = num_nodes
    Hyperparm.shapeY = shapeY
    Hyperparm.shapeX = shapeX
    Hyperparm.pre_trained_model = pre_trained_model
    Hyperparm.last_layer_output = last_layer_output
    
    return Hyperparm

def build_model_RMSprop(pre_trained_model, cnn_last_output,Dropout = False, dropout_rate = 0.5, learning_rate = 0.00001, hidden_units_num = 1024, num_classes = 1, activation='sigmoid'):

  # Flatten the output layer to 1 dimension
  x = layers.Flatten()(cnn_last_output)
  # Add a fully connected layer with 1,024 hidden units and ReLU activation
  x = layers.Dense(hidden_units_num, activation='relu')(x)
  # Add a dropout rate of 0.2
  if Dropout:
      x = layers.Dropout(dropout_rate)(x)
  # Add a final sigmoid layer for classification
  x = layers.Dense(num_classes, activation=activation)(x)

  # Configure and compile the model
  model = Model(pre_trained_model.input, x)
  model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(lr=learning_rate), #Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1, amsgrad=False)
                metrics=['acc'])
  return model


def runModel(Partitioned_path,Genotypes,Hyperparameters,epochs = 2):
    #Save partitions to respective folder
    saveFiles(Genotypes)
    
    # Creating training and validation data generators from separated data
    # Transformations are for training generator only
    
    #Delay 15 seconds for syncing purposes
    train_generator, validation_generator = create_data_generators(
        Partitioned_path,
        Hyperparameters.shapeY, Hyperparameters.shapeX, 
        train_batch_size=len(Genotypes[0].trainSet), 
        validation_batch_size=len(Genotypes[0].testSet),
        class_mode='categorical',
        horizontal_flip = False)
    
    #Delay 15 seconds for syncing purposes
    time.sleep(15)
        
    # Building model
    model = build_model_RMSprop(Hyperparameters.pre_trained_model, Hyperparameters.last_layer_output, 
                            Dropout = False,
                            dropout_rate = Hyperparameters.dropout_rate, 
                            learning_rate = Hyperparameters.LR ,
                            hidden_units_num = Hyperparameters.num_nodes,
                            num_classes=len(Genotypes),
                            activation = 'softmax')
    
    # Training model
    history = model.fit_generator(
      train_generator,
      steps_per_epoch=3,
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps=3,
      verbose=2)
    
    return Genotypes,history,model


def create_data_generators(data_src_path, shapeY, shapeX, train_batch_size = 5, validation_batch_size = 10, rotation_range = 0, width_shift_range = 0, height_shift_range = 0, shear_range = 0, zoom_range = 0, horizontal_flip = False, class_mode = 'binary'):

  # Define our train and validation directories and files
  train_dir = os.path.join(data_src_path, 'Train')
  validation_dir = os.path.join(data_src_path, 'Validation')



  # Add our data-augmentation parameters to ImageDataGenerator
  train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=rotation_range,
      width_shift_range=width_shift_range,
      height_shift_range=height_shift_range,
      shear_range=shear_range,
      zoom_range=zoom_range,
      horizontal_flip=horizontal_flip)

  # Note that the validation data should not be augmented!
  validation_datagen = ImageDataGenerator(rescale=1./255)

  train_generator = train_datagen.flow_from_directory(
          train_dir, # This is the source directory for training images
          target_size=(shapeY, shapeX),  # All images will be resized to 150x150
          batch_size=train_batch_size,
          # Since we use binary_crossentropy loss, we need binary labels
          class_mode=class_mode)

  # Flow validation images in batches of 20 using test_datagen generator
  validation_generator = validation_datagen.flow_from_directory(
          validation_dir,
          target_size=(shapeY, shapeX),
          batch_size=validation_batch_size,
          class_mode=class_mode)
  
  return train_generator, validation_generator