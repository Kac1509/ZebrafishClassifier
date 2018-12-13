from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop, Adam

def build_model_RMSprop(pre_trained_model, cnn_last_output, learning_rate = 0.00001, hidden_units_num = 1024,num_classes = 1,activation='sigmoid'):

  # Flatten the output layer to 1 dimension
  x = layers.Flatten()(cnn_last_output)
  # Add a fully connected layer with 1,024 hidden units and ReLU activation
  x = layers.Dense(hidden_units_num, activation='relu')(x)
  # Add a dropout rate of 0.2
  #x = layers.Dropout(0.2)(x)
  # Add a final sigmoid layer for classification
  x = layers.Dense(num_classes, activation=activation)(x)

  # Configure and compile the model
  model = Model(pre_trained_model.input, x)
  model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(lr=learning_rate), #Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1, amsgrad=False)
                metrics=['acc'])
  return model