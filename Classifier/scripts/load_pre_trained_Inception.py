from tensorflow.keras.applications.inception_v3 import InceptionV3

def load_pre_trained_Inception(path, shapeY, shapeX, color_channels):
  
  pre_trained_model = InceptionV3(
    input_shape=(shapeY, shapeX, color_channels), include_top=False, weights=None)
  
  # loading pre-trained weights
  pre_trained_model.load_weights(path)
  
  # making model non-trainable
  for layer in pre_trained_model.layers:
    layer.trainable = False
  
  # chosing cnn output layer
  last_layer = pre_trained_model.get_layer('mixed7')
  #print 'last layer output shape:', last_layer.output_shape
  
  return pre_trained_model, last_layer.output