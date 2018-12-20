# Zebrafish Classifier

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
with Python 3.6. The next step is to install of the required prerequist through the Anaconda
prompt, i.e. "conda install numpy" 

The project can be ran in the Jupyter Notebook environment, Google Colaboratory environment or
directly through the command line. Google Colaboratory was used for computationally depending
tasks leveraging their provided GPU. It is was particularly useful for running our model with
bigger input sizes for the images. 

To run from the command line, open the command line in the directory where the ZebrafishClassifier 
folder is located. Then simply paste the following command:
python "directory_path\ZebrafishClassifier\CNN_Classifier\scripts\run.py"

To run from the Jupyter Notebook environment, simply open and run through each cell:
directory_path\ZebrafishClassifier\CNN_Classifier.ipynb

To run from the Colaboratory environment, open the link below and follow these steps:
  - Create the following directory in the root of Google Drive 'Colab Notebooks/CNN_Classifier/scripts'
  - Add all .py files in the scripts folder
  - Add 'DataStraightened.zip' to Colab Notebooks/CNN_Classifier/
  - Set Colab to True
  - You will be prompted to give Google Colaboratory access to Google Drive where the scripts and data
  files are located
https://colab.research.google.com/github/Kac1509/ZebrafishClassifier/blob/master/ClassifierFinal.ipynb

Running through the command line does not allow for data visualization. For this, it is recommended
to use Jupyter or Google Colaboratory. Furthermore, the latter two environments allow the user to access
the code. This is important for chaning the parameters and running the hyperparameter tuning program.

### Models

- K-nearest neighbors
- VGG16 CNN with one hidden layer
- InceptionV3 CNN with one hidden layer

### Main Program
The main program is run.py(). Executing this program will train our best image classifier model using
training data from the 'DataStraightened.zip' file. After training the model, the program will 
predict the genotypes of the images placed in the Predictions folder. In this case, this represents
our test data.

As a default, the model being trained uses a pre-trained CNN network with a fully-connected output
layer. We defined the hyperparameters for our model to be the input size, number of hidden nodes, 
dropout rate and learning rate. The optimal values for our model were determined to be: input size
set to 75x375, number of hidden nodes set to 256,dropout rate set to 0.5 and the learning rate for
the RMSProp optimizer set to 10E-4. 

This program is quite computationally intensive and may take a few minutes to run. Faster performance
can be obtained by running the program in the Google Colaboratory. The optimal model presented above
is best ran in the Google Colaboratory environment

### Hyperparameter tuning
Hyperparameter tuning requires access to the code, therefore it can be done using the Jupyter Notebook
or the Google Colaboratory environment. Hyperparameter tuning is conducted using K-fold cross-validation.
The pipeline to tune a particular parameter is the same for all parameters, except for input size. However,
it is necessary to set the given parameter under test manually (commenting and uncommenting code). For 
input size, Input Size boolean must be set to True in the Hyperparameter_tuning function. Hyperparameter 
tuning is a combination of grid search and cross-validation therefore it can be quite computationally 
laborious. Therefore, it is recommended to perform this task in the Google Colaboratory environment.


### Folder Structure
├── ZebrafishClassifier
  ├── CNN_Classifier                          # CNN Model files 
  │   ├── scripts                             # Script Files
  │   │   ├── CNN_Model.py
  │   │   ├── DataVisualization.py
  │   │   ├── Genotype.py
  │   │   ├── Helpers.py
  │   │   ├── Pre_trained_models.py
  │   │   ├── Predictions.py
  │   │   ├── Setup.py
  │   ├── weights                             # Weight Files
  │   │   ├── inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
  │   │   ├── vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
  │   ├── Extracted_data                      # Extracted Files (Only created once the main program is executed)
  │   ├── PartitionedData                     # Partioned Files (Only created once the main program is executed)
  │   ├── Predictions                         # Prediction Files
  │   │   ├── Straightened
  │   │   │   ├── test-s.png
       ...
  │   │   │   ├── test50-s.png 
  ├── KNN_Classifier                           # KNN Model files
  │   ├── scripts                              # Script Files
  │   │   ├── KNN_helpers.py
  │   │   ├── Setup.py
  │   ├── Extracted_data                       # Extracted Files (Only created once the main program is executed)
  │   ├── KNN.ipynb							   # Notebook that shows and reproduces KNN results
  │   ├── main_knn.py						   
  ├── README.MD                                # Readme file 
  ├── Requirements.txt                         # Requirements 
  ├── Results                                  # Results of the test data using our best model
  │   ├──Results_Option1_2018-12-19_1027.csv   #Results from test set 1
  │   ├──Results_Option2_2018-12-19_1034.csv   #Results from test set 1


Some of these folder are generated once the main program is executed

### Scripts Structure (.py files)
├── CNN_Model.py
This file contains all of the required functions for generating our CNN model. The parameters
of the model are defined and set here. These user-defined parameters consists of the input size, 
dropout rate, learning rate, number of nodes, pre-trained model, activation function, loss 
function and optimizer. Model configuration, compilation and execution is located in this file
Each time a model is trained, the paritions for each genotype is first saved into their respective
train and validation folders. The main function for hyperparameter tuning using cross-validation
is located in this file, see the section above for further detail.

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

├── main_knn.py
This file contains all of the required functions for generating and evaluating our KNN model. The parameters
of the model are defined and set here. Model configuration, compilation and execution is located in this file.

├── KNN_helpers.py
All helper functions regarding KNN model are located here. It consists primairly of functions for file and folder
manipulation.

## Authors

* **Kaleem Corbin** - (https://github.com/Kac1509)
* **Lazar Stojkovic** - (https://github.com/stojk)
* **Vojislav Gligorovski** - (https://github.com/voja95)
