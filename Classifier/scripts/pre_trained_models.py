from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3

def load_pre_trained_Inception(path, Parameters, color_channels):
  
  pre_trained_model = InceptionV3(
    input_shape=(Parameters.shapeY, Parameters.shapeX, color_channels), include_top=False, weights=None)
  
  # loading pre-trained weights
  pre_trained_model.load_weights(path)
  
  # making model non-trainable
  for layer in pre_trained_model.layers:
    layer.trainable = False
  
  # chosing cnn output layer
  last_layer = pre_trained_model.get_layer('mixed7')
  #print 'last layer output shape:', last_layer.output_shape
  
  return pre_trained_model, last_layer.output


def load_pre_trained_VGG16(path, Parameters, color_channels):
  
  pre_trained_model = VGG16(
    input_shape=(Parameters.shapeY, Parameters.shapeX, color_channels), include_top=False, weights=None)
  
  # loading pre-trained weights
  pre_trained_model.load_weights(path)
  
  # making model non-trainable
  for layer in pre_trained_model.layers:
    layer.trainable = False
  
  # chosing cnn output layer
  last_layer = pre_trained_model.get_layer('block5_pool')
  #print 'last layer output shape:', last_layer.output_shape
  
  return pre_trained_model, last_layer.output