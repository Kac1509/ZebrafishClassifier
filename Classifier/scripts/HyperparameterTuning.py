from sklearn.model_selection import KFold
from operator import itemgetter
import numpy as np
from Helpers import *
from CNN_Model import *
from Genotype import *
import time



def CV_run(Genotypes_Fold,History_Fold,Model_Fold,Parameters,Genotypes,Paths,epochs,n_splits):
    
    # Retrieve size of genotype with fewest images
    idx,size = retrieveSmall(Genotypes,sys.maxsize)
    
    #K-fold validation splitting
    kf = KFold(n_splits=n_splits, shuffle=True,random_state = 7)
    kf.get_n_splits(Genotypes[idx].images) 
    Fold = 0;
 
    for train_index, test_index in kf.split(Genotypes[idx].images):
        
        #Added a delay for syncing purposes (Particularly for cloud storage)
        print("Fold: ", Fold," test_index: ",test_index)
        time.sleep(15)
        createDirectories(len(Genotypes),Paths,Genotypes)
        
        for i in range(len(Genotypes)):
            Genotypes[i].trainSet = itemgetter(*train_index)(Genotypes[i].images)
            Genotypes[i].testSet = itemgetter(*test_index)(Genotypes[i].images)
                        
        History,Model = runModel(Paths,Genotypes,Parameters,epochs)
        Genotypes_Fold.append(Genotypes),History_Fold.append(History),Model_Fold.append(Model)         
        Fold+=1

def Hyperparameter_tuning(Paths,Genotypes,Parameters,Hyperparameter,epochs,k_fold):
      
    #Set true if the hyperparameter is Input_Size
    Input_Size = False
   
    #Store train & test mean and standard deviation loss 
    loss_tr = []
    loss_te = []
    loss_trSTD = []
    loss_teSTD = []   
 
    #Iterate through values of hyperparmeter being tuned
    for parameter in Hyperparameter:
        print(Hyperparameter)
        print('parameter = ' + str(parameter))    
        Genotypes_Fold = []
        History_Fold = []
        Model_Fold = []
        if Input_Size:
            Parameters.local_weights_file_VGG16 = Paths.base_path + 'weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
            Parameters.pre_trained_model_VGG16, Parameters.last_layer_output_VGG16 = load_pre_trained_VGG16(local_weights_file_VGG16,
                                                                                                            Parameters,
                                                                                                            color_channels)
            
        #cross validation
        CV_run(Genotypes_Fold,History_Fold,Model_Fold,
               Parameters,
               Genotypes,
               Paths,
               epochs,
               n_splits = k_fold)
        
        #Store loss for each fold
        loss_tr_tmp = []
        loss_te_tmp = []
        for i in range(k_fold):
          #plot_loss(history, lr)
          loss_tr_tmp.append(History_Fold[i].history['loss'][-1])
          loss_te_tmp.append(History_Fold[i].history['val_loss'][-1])
        loss_tr.append(np.mean(loss_tr_tmp))
        loss_te.append(np.mean(loss_te_tmp))
        loss_trSTD.append(np.std(loss_tr_tmp))
        loss_teSTD.append(np.std(loss_te_tmp))
        
        
    loss_tr = np.asarray(loss_tr)
    loss_te = np.asarray(loss_te)
    loss_trSTD = np.asarray(loss_trSTD)
    loss_teSTD = np.asarray(loss_teSTD)
    
    Losses = [ np.asarray(loss_tr),np.asarray(loss_te),np.asarray(loss_trSTD),np.asarray(loss_teSTD)]
    
    return Hyperparameter,Losses


        
        