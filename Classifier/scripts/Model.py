from saveFiles import *
from create_data_generators import *
from build_model_RMSprop import *
import time

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