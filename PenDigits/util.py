import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import sklearn.metrics
from RandomSignature import *

## taken from git repository of 'Anomaly detection in streamed data'
def load_pendigits_dataset(filename):
    with open(filename, 'r') as f:
        data_lines = f.readlines()

    data = []
    data_labels = []
    current_digit = None

    for line in data_lines:
        if line == "\n":
            continue

        if line[0] == ".":
            if "SEGMENT DIGIT" in line[1:]:
                if current_digit is not None:
                    data.append(np.array(current_digit))
                    data_labels.append(digit_label)

                current_digit = []
                digit_label = int(line.split('"')[1])
            else:
                continue

        else:
            x, y = map(float, line.split())
            current_digit.append([x, y])
            
    data.append(np.array(current_digit))
    data_labels.append(digit_label)

    return data, data_labels

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
    digits_correct_pred = []
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
        digits_correct_pred.append(pred == true)
        if verbose:
            print('ROC AUC for digit ',digit,': ', ROC_AUC,'\n')
        mean_ROC_AUC += ROC_AUC/10

    if verbose:    
        print('Mean accuracy: ',mean_accuracy)
        print('Mean ROC AUC: ', mean_ROC_AUC)
    
    return mean_accuracy, mean_ROC_AUC, digits_accuracy, digits_ROC_AUC, digits_correct_pred

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
    plt.title('Empirical cumulative distribution of the Isolation Score \n (Method: {}, Reservoir dimension: k = {})'.format(
        clf[1],hparams['res_size']))
    if save:
        plt.savefig('empirical_cdf_res_size_{}.pdf'.format(hparams['res_size']))
    plt.show()
