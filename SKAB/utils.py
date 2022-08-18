import os
import pandas as pd
import sys
sys.path.append('../PenDigitsRanSig/')
import sklearn
from RandomSignature import *
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


def read_SKAB(anomaly_free = False):
# open the .csv files from the SKAB datasets and return a list of pandas df
# anomaly_free = True if we also want to load the anomaly-free run. The df is then
# converted to numpy and returned separately
  
    all_files=[]
    for root, dirs, files in os.walk("../Data/SKAB_data/"):
        for file in files:
            if file.endswith(".csv"):
                 all_files.append(os.path.join(root, file))

    list_of_df = [pd.read_csv(file, 
                              sep=';', 
                              parse_dates=True) for file in all_files if 'anomaly-free' not in file]
    if anomaly_free:
        path_anomaly_free = '../Data/SKAB_data/anomaly-free/anomaly-free.csv'
        df_anomaly_free = pd.read_csv(path_anomaly_free,sep = ';')
        df_anomaly_free = df_anomaly_free.drop(columns = ['datetime'])
        paths_anomaly_free = df_anomaly_free.to_numpy()
        return list_of_df, paths_anomaly_free
    else:
        return list_of_df
        

def evaluate(hyperparams_dict,len_sub,clf,cont):
## evaluate the predictions on all the different runs as done in the 
## demo notebook.
## hyperparams_dict: dictionary containing the parameters for the rsig computation
## len_sub: parameter l of the window length for the equidistant splitting
## clf: str containing name of model to be used for predictions 
## 'SVM' for one class SVM and 'IF' for isolation forest
## cont: contamination parameter for the clf
  
    dim_paths = 8

    [As,bs] = get_random_coeff(dim_paths,hyperparams_dict)

    ## test the model 
    ## read the data
    list_of_df = util.read_SKAB(anomaly_free = False)


    anomalies = []
    predictions = []
    
    for df_test in list_of_df:
        #values needed to evaluate the performance
        #save values for anomaly and changepoints
        anomalies_temp = df_test['anomaly'].to_list()#[400:]
        anomalies = anomalies + anomalies_temp
        
        # final paths
        df_test = df_test.drop(columns = ['datetime','anomaly','changepoint'])
        path = df_test.to_numpy()
        
        ## scale (pre)
        path = sklearn.preprocessing.MinMaxScaler().fit_transform(path)
        


        #split in train and test and prepare the subsequences
        path_train = path[:400,:]
        path_test = path[400:,:]
        
        
        ## split into sub-series
        path_train = [path_train[len_sub*i:len_sub*(i+1),:] for i in range(int(path_train.shape[0]/len_sub+1))]
        #if the 400 or the time series is an exact multiple of len_sub then the last element in paths_train is [] 
        if len(path_train[-1])==0:
            path_train.pop()
        path_test = [path_test[len_sub*i:len_sub*(i+1),:] for i in range(int(path_test.shape[0]/len_sub+1))]
        if len(path_test[-1])==0:
            path_test.pop()   
        
        ## augmentations
        # basepoint
        path_train = [np.vstack((np.zeros_like(path[0,:]),path)) for path in path_train]
        path_test = [np.vstack((np.zeros_like(path[0,:]),path)) for path in path_test]
        
        
        ## compute the signature for train and test paths
        Sigs_train = [compute_signature(As,bs,path_train[i],False,hyperparams_dict) 
                         for i in range(len(path_train))]
        Sigs_test = [compute_signature(As,bs,path_test[i],False,hyperparams_dict) 
                         for i in range(len(path_test))]
        
        path = path_train+path_test
        Sigs = np.array(Sigs_train+Sigs_test)
        Sigs_train = np.array(Sigs_train)
        Sigs_test = np.array(Sigs_test)

        ## fit the model
        if clf == 'IF':
            clf = IsolationForest(contamination = cont)
        elif clf == 'SVM':
            clf = OneClassSVM(nu = cont)
        
        clf.fit(Sigs_train)
        
        # store results as pd series
        pred_anomalies = []
        for i in range(Sigs.shape[0]):
            ## in np.ones() the -1 is needed because with the base point augmentation
            ## we make the sub-series one observation longer
            pred_anomalies = pred_anomalies + (-0.5*(clf.predict(Sigs)[i]-np.ones(path[i].shape[0]-1))).tolist()
        predictions = predictions + pred_anomalies
                


    FAR = (np.array(predictions)[np.array(anomalies) == 0] != np.array(anomalies)[np.array(anomalies) == 0]).sum()/np.array(anomalies)[np.array(anomalies) == 0].shape[0]
    MAR = (np.array(predictions)[np.array(anomalies) == 1] != np.array(anomalies)[np.array(anomalies) == 1]).sum()/np.array(anomalies)[np.array(anomalies) == 1].shape[0]
    F1 = sklearn.metrics.f1_score(anomalies,predictions)
    

    
    return  F1, FAR, MAR