# ZebrafishClassifier

The object of this project was to construct a highly reliable image classification technique
for distinguishing between fss, gullum and wild type zebrafish embryos.

### Prerequisites

- Python 3.6
- Matplotlib 2.0.2
- Tensorflow 1.12.0
- Numpy 1.15.4


### Installing - Programming Environment

The first step is to setup the Python envirnoment.

The easiest way to set this up and acquire all the necessary dependencies is to install Anaconda
with Python 3.6

The project can be ran in the Jupyter Notebook environment. We also used Google Colaboratory for 
computationally depending tasks leveraging their provided GPU. 


### Models

- K-nearest neighbors
- VGG16 CNN with one hidden layer
- InceptionV3 CNN with one hidden layer

### Main Program
The main program is run.py(). Executing this program will train our best image classifier using
training data from the 'DataStraightened.zip' file. After training the model, the program will 
predict the genotypes of the images placed in the Predictions folder. In this case, this represents
our test data.

As a default, the model being trained uses a pre-trained CNN network with a fully-connected output
layer. We defined the hyperparameters for our model to be the input size, number of hidden nodes, 
dropout rate and learning rate. The optimal values for our model were determined to be: input size
set to 75x375, number of hidden nodes set to 256,dropout rate set to 0.5 and the learning rate for
the RMSProp optimizer set to 10E-4.



### Folder Structure
├── ZebrafishClassifier
  ├── CNN_Classifier                    # CNN Model files 
  │   ├── scripts                       # Script Files
  │   │   ├── CNN_Model.py
  │   │   ├── DataVisualization.py
  │   │   ├── Genotype.py
  │   │   ├── Helpers.py
  │   │   ├── Pre_trained_models.py
  │   │   ├── Predictions.py
  │   │   ├── Setup.py
  │   ├── weights                       # Weight Files
  │   │   ├── inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
  │   │   ├── vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
  │   ├── Extracted_data                # Extracted Files (Only created once the main program is executed)
  │   ├── PartitionedData               # Partioned Files (Only created once the main program is executed)
  │   ├── Predictions                   # Prediction Files
  │   │   ├── Straightened
  │   │   │   ├── test-s.png
       ...
  │   │   │   ├── test50-s.png 
  ├── KNN_Classifier                    # KNN Model files
  │   ├── benchmarks
  ├── README.MD                         # Readme file 
  ├── Requirements.txt                  # Requirements 


Some of these folder are generated once the main program is executed

### Scripts Structure (.py files)
├── CNN_Model.py
This file contains all of the required functions for generating our CNN model. The parameters
of the model are defined and set here. These user-defined parameters consists of the input size, 
dropout rate, learning rate, number of nodes, pre-trained model, activation function, loss 
function and optimizer. Model configuration, compilation and execution is located in this file
Each time a model is trained, the paritions for each genotype is first saved into their respective
train and validation folders.

├── DataVisualization.py
All data visulatization and plotting functions are located in this file. First plot is to
visualize the train & test accuracy and error. Second plot is used to visualize the tuning of
the hyperparameters using cross-validation. It plots the mean and standard deviation for each
parameter evaluated.

├── Genotype.py
The Genotype class is defined in this class and contains genotype related information. It also
contains the function responsible for partitioning the data into train and test sets (when not 
using cross-validation).

├── Helpers.py
All helper functions are located here. It consists primairly of functions for file and folder
manipulation.

├── Pre_trained_models.py
All parameters for the pre-trained model are located here. The process of extracting the bottleneck 
layer for both VGG16 and InceptionV3 is located in this file

├── Predictions.py
Prediction functions are located here. The predictClass function predicts the class for the images
located in the Predictions folder. It also contains a function to save the predictions and model
parameters to a csv file.

├── Setup.py
This file is used to set up the environment. In this file all the paths are set and the data is 
extracted from the zip file. All folders are cleared to reset the environment for the next run.

## Authors

* **Kaleem Corbin** - (https://github.com/Kac1509)
* **Lazar Stojkovic** - (https://github.com/stojk)
* **Vojislav Gligorovski** - (https://github.com/voja95)
