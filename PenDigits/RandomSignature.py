import numpy as np
import pandas as pd
import util 
import sklearn.metrics
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))

hyperparams_dict = {
'varA':0.3,
'mean':0,
'res_size':10,
'activation': sigmoid
}

def get_random_coeff(d,hparams = hyperparams_dict):
# hyperparam_dict: contains mean and var of normal 
# distribution that we want to sample as well as dimension
# of randomized signature.
# d: dimension of the control, and therefore number 
# of different matrix,vector pairs of coeff to be generated
# output: tuple(As,bs), with As np array (d,res_size,res_size)
# and bs np array (d,res_size). As[i,:,:] and bs[i,:] 
# are the i-th components of the vector field generating the 
# rand signature.
 
    random_projection = []
    random_bias = []
    for i in range(d):


        projection = np.random.normal(hparams['mean'], hparams['varA'],
            (hparams['res_size'], hparams['res_size']))
        
        norm = np.linalg.norm(projection, 2)
        #why are wenormalizing to 0.99?
        projection = projection / norm * 0.99
        random_projection.append(projection)

        random_bias.append(np.random.normal(hparams['mean'], hparams['varA'], size=hparams['res_size']))

    return np.array(random_projection), np.array(random_bias)



def compute_signature(As,bs,path,trajectory = False,hparams = hyperparams_dict):
# Solve the IVP given by r-Sig_0 = (1,...,1) (may need to change this) and 
# dr-Sig = sigma(As[i,:,:]r-Sig+b[i,:])dpath[i,:]. Working for a single path, that is
# if we have a batch of paths we have to call this function once for each path.
# As,bs,path: coeffeicient of the above ODE
# trajectory: bool, if False only return the value of r-Sig_T, where the T
# is the final time of path  (i.e. len(path[0])). if True, then the whole
# trajectory r-Sig_t t=0,...,T is returned.
# hparams: dictionary of parameters containing activation function (may be replaced
# with just passing the activation function) 
# output: if trajectory is False, return vector of length As.shape[1] corresponding to r-Sig_T.
# If trajectory is True, return matrix (r-Sig_0,...,r-Sig_T) of shape (len(dX)+1,As.shape[1])

    dX = np.diff(path,axis = 0)
    
    if trajectory:
        Sig = np.zeros((len(dX)+1,As.shape[1]))
        Sig[0,:] = np.ones(As.shape[1])
        for i in range(len(dX)):
            Sig[1+i,:] = Sig[i,:]+hparams['activation'](np.dot(As, Sig[i,:]) + bs).T.dot(dX[i])
    else:
        Sig = np.ones(As.shape[1])
        for i in range(len(dX)):
            dSig = hparams['activation'](np.dot(As, Sig) + bs).T.dot(dX[i])
            Sig += dSig
            
    return Sig
    
def get_signature(paths,trajectory = False,**kwargs):
#Return the Signature for all the path in paths.
#paths: batch of paths for which we want to compute the Signature.
#trajectory: see above (function compute_signature)
#output: depending on trajectory, ndarray of shape [len(paths),dim(r-Sig)], where
# the dimension of r-Sig depends on the value of trajectory as described above.
   
    # paths dimension
    paths_dim = paths[0].shape[1]
    # generate vector fields
    [As,bs] = get_random_coeff(d = paths_dim,**kwargs)

    #Compute Signature for all observations
    Sigs = np.array([compute_signature(As,bs,paths[i],trajectory,**kwargs) 
                     for i in range(len(paths))])
    return Sigs

def evaluate_over_digits(Sigs,df,clf,verbose = False):
# compute mean accuracy and mean ROC_AUC for model clf.
# Sigs: value obtained with get_signature with trajectory = False. That is
# a batch of features vectors based on which we want to detect anomalies
# df: original data-set as pandas df, needed to get the position of the
# instances of various classes.
# clf: sklearn model that we want to evaluate. May need to change the code
# to use different models. Until now isolation forest and one class svm used.
# output: mean_accuracy, mean_ROC_AUC, digits_accuracy, digits_ROC_AUC, where
# mean_accuracy, mean_ROC_AUC are the mean value taken over all digits and 
# digits_accuracy, digits_ROC_AUC are lists where the i-th value of the list
# corresponds to the accuracy/ROC_AUC obtained for digit i.
    
    mean_accuracy = 0
    mean_ROC_AUC = 0
    digits_accuracy = []
    digits_ROC_AUC = []
    for digit in range(10):

        X_train = Sigs[(df['Digit'] == digit) & (df['Subset'] == 'train'),:]
        X_test = Sigs[df['Subset'] == 'test',:]

        # fit the model
        #clf = IsolationForest(max_samples=100)
        #clf = OneClassSVM(nu = 0.7)
        
        clf.fit(X_train)
        y_pred_outliers = clf.predict(X_test)

        # evaluate the model
        pred = y_pred_outliers == 1
        true = df[df['Subset']== 'test']['Digit'] == digit
        accuracy = np.mean(pred == true)
        digits_accuracy.append(accuracy)
        if verbose:
            print('Accuracy for digit ',digit,': ', accuracy)
        
        mean_accuracy += accuracy/10
        
        ## roc auc score
        ROC_AUC = sklearn.metrics.roc_auc_score(true,clf.decision_function(X_test))
        digits_ROC_AUC.append(ROC_AUC)
        
        if verbose:
            print('ROC AUC for digit ',digit,': ', ROC_AUC,'\n')
        mean_ROC_AUC += ROC_AUC/10

    if verbose:    
        print('Mean accuracy: ',mean_accuracy)
        print('Mean ROC AUC: ', mean_ROC_AUC)
    
    return mean_accuracy, mean_ROC_AUC, digits_accuracy, digits_ROC_AUC

def ecdf(x):
#compute the empirical cumulative distribution function for the values in x.
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys   


def plot_score(paths,df,clf,hparams, save = False):
# compute and plot the ecdf (same as in Anomaly detection on Streamed data)
# of Anomaly score (computed with clf and r-Sig constructed with parameters in hparams)
# for normal and anomalous observations.
# paths: batch of paths used to compute r-Sigs
# df: original df used to locate instances of the different calsses as well as train/test set
# clf: model used to compute anomaly Score
# hparams: parameters used to generate the r-Sig
# save: bool, if we want to save the plot or not.
    
    Sigs = get_signature(paths,hparams = hparams)
    inliers = []
    outliers = []
    
    for digit in range(10):
        
        X_train = Sigs[(df['Digit'] == digit) & (df['Subset'] == 'train'),:]
        
        X_test_inliers = Sigs[(df['Digit'] == digit) & (df['Subset'] == 'test'),:]
        X_test_outliers = Sigs[(df['Digit'] != digit) & (df['Subset'] == 'test'),:]
        
        clf[0].fit(X_train)
        inliers += clf[0].decision_function(X_test_inliers).tolist()
        outliers += clf[0].decision_function(X_test_outliers).tolist()
    
    xs, ys = ecdf(inliers)
    plt.plot(xs, ys, label='Normal', linestyle='--', linewidth=3)
    xs, ys = ecdf(outliers)
    plt.plot(xs, ys, label='Anomalous', linestyle='-', linewidth=3)
    plt.xlabel('Isolation Score')
    plt.ylabel('Cumulative probability')
    plt.legend(loc='lower right')
    plt.grid()
    plt.title('Empirical cumulative distribution of the Isolation Score \n (Method: {}, Reservoir dimension: N = {})'.format(
        clf[1],hparams['res_size']))
    if save:
        plt.savefig('empirical_cdf_res_size_{}.pdf'.format(hparams['res_size']))
    plt.show()    
 