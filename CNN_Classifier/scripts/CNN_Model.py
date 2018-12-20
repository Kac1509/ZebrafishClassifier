from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Pre_trained_models import *
from Helpers import *

import os
import time

class Parameters:
    def __init__(self,Dropout):
        self.Dropout = Dropout

def setParameters(Paths, shapeY = 50, shapeX = 50, dropout_rate = 0, LR = 0.0001, num_nodes = 256, VGG16 = True,Dropout = True):
    
    
    Parm = Parameters(Dropout)
    Parm.dropout_rate = dropout_rate
    Parm.LR = LR
    Parm.num_nodes = num_nodes
    Parm.shapeY = shapeY
    Parm.shapeX = shapeX
   
    
    if VGG16:
        #VGG16 Model
        local_weights_file_VGG16 = Paths.base_path + 'weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        pre_trained_model, last_layer_output = load_pre_trained_VGG16(local_weights_file_VGG16, Parm, color_channels = 3)
    else:
        # Inception Model
        local_weights_file_Inception = Paths.base_path + 'weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        pre_trained_model, last_layer_output = load_pre_trained_Inception(local_weights_file_Inception, Parm, color_channels = 3)
        
    Parm.pre_trained_model = pre_trained_model
    Parm.last_layer_output = last_layer_output

    
    return Parm

def build_model_RMSprop(Parameters, num_classes, activation='sigmoid'):

  # Flatten the output layer to 1 dimension
  x = layers.Flatten()(Parameters.last_layer_output)
  # Add a fully connected layer with 1,024 hidden units and ReLU activation
  x = layers.Dense(Parameters.num_nodes, activation='relu')(x)
  # Add a dropout rate of 0.2
  if Parameters.Dropout:
      x = layers.Dropout(Parameters.dropout_rate)(x)
  # Add a final sigmoid layer for classification
  x = layers.Dense(num_classes, activation=activation)(x)

  # Configure and compile the model
  model = Model(Parameters.pre_trained_model.input, x)
  model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(lr=Parameters.LR), #Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1, amsgrad=False)
                metrics=['acc'])
  return model


def runModel(Paths,Genotypes,Parameters,epochs = 2):
    #Save partitions to respective folder
    saveFiles(Genotypes)
    
    # Creating training and validation data generators from separated data
    # Transformations are for training generator only
    
    #Delay for syncing purposes
    time.sleep(1)
    train_generator, validation_generator = create_data_generators(
        Paths.partitioned_path,
        Parameters, 
        train_batch_size=len(Genotypes[0].trainSet), 
        validation_batch_size=len(Genotypes[0].testSet),
        class_mode='categorical',
        horizontal_flip = False)
    
    #Delay for syncing purposes
    time.sleep(1)
        
    # Building model
    model = build_model_RMSprop(Parameters,num_classes=len(Genotypes),activation = 'softmax')
    
    # Training model
    history = model.fit_generator(
      train_generator,
      steps_per_epoch=len(Genotypes),
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps=len(Genotypes),
      verbose=2)
    
    return history,model


def create_data_generators(data_src_path, Parameters, train_batch_size = 5, validation_batch_size = 10, rotation_range = 0, width_shift_range = 0, height_shift_range = 0, shear_range = 0, zoom_range = 0, horizontal_flip = False, class_mode = 'binary'):

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
          train_dir,
          target_size=(Parameters.shapeY, Parameters.shapeX),  # All images are resized 
          batch_size=train_batch_size,
          # Since we use binary_crossentropy loss, we need binary labels
          class_mode=class_mode)

  validation_generator = validation_datagen.flow_from_directory(
          validation_dir,
          target_size=(Parameters.shapeY, Parameters.shapeX),  # All images are resized 
          batch_size=validation_batch_size,
          class_mode=class_mode)
  
  return train_generator, validation_generator