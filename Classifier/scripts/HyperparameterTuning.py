from sklearn.model_selection import KFold
from operator import itemgetter
import numpy as np
from Helpers import *
from CNN_Model import *
from Genotype import *



def CV_run(Genotypes_Fold,History_Fold,Model_Fold,Hyperparameters,Genotypes,
               Extracted_path,
               Partitioned_path,epochs,n_splits):
    idx,size = retrieveSmall(Genotypes,sys.maxsize)
    kf = KFold(n_splits=n_splits, shuffle=True,random_state = 7)
    kf.get_n_splits(Genotypes[idx].images) 
    Fold = 0;
 
    for train_index, test_index in kf.split(Genotypes[idx].images):

        print("Fold: ", Fold," test_index: ",test_index)
        createDirectories(len(Genotypes),Extracted_path,Partitioned_path,Genotypes)

        for i in range(len(Genotypes)):
            Genotypes[i].trainSet = itemgetter(*train_index)(Genotypes[i].images)
            Genotypes[i].testSet = itemgetter(*test_index)(Genotypes[i].images)
            #print("Train_set size: ", len(Genotypes[i].trainSet), " Test_set size: ",len(Genotypes[i].testSet))
            
        Gen,Hist,Mdl = runModel(Partitioned_path,Genotypes,Hyperparameters,epochs)
        Genotypes_Fold.append(Gen),History_Fold.append(Hist),Model_Fold.append(Mdl) 
        
        Fold+=1

def Hyperparameter_tuning(Base_path,Extracted_path,Partitioned_path,Genotypes,Hyperparameters,HypVals):

    k_fold = 3
    
   
    # split data in k fold
    # k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    loss_tr = []
    loss_te = []
    loss_trSTD = []
    loss_teSTD = []
    
    IS = False
    
    # cross validation
    for parameter in HypVals:
        
        
        print(HypVals)
        print('parameter = ' + str(parameter))    
        Genotypes_Fold = []
        History_Fold = []
        Model_Fold = []
        if IS:
            Hyperparameters.local_weights_file_VGG16 = Base_path + 'weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
            Hyperparameters.pre_trained_model_VGG16, Hyperparameters.last_layer_output_VGG16 = load_pre_trained_VGG16(local_weights_file_VGG16, Hyperparameters.shapeY, Hyperparameters.shapeX, color_channels)
            
        
    
        CV_run(Genotypes_Fold,History_Fold,Model_Fold,
               Hyperparameters,
               Genotypes,
               Extracted_path,
               Partitioned_path,
               epochs=3,
               n_splits = k_fold)
        
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
    
    return HypVals,Losses


        
        