from load_pre_trained_VGG16 import *
from load_pre_trained_Inception import *


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