## create an heatmap showing how the different
## combinations of parameters N and sigma_A
## affect the performance of the model.
import util 
import pandas as pd
import numpy as np
import sklearn.preprocessing
from RandomSignature import *
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

data = {'train': util.load_pendigits_dataset('../Data/PenDigits_Data/pendigits-orig.tra'),
        'test': util.load_pendigits_dataset('../Data/PenDigits_Data/pendigits-orig.tes')}

dataframes = []
for subset, data in data.items():
    df = pd.DataFrame(data).T
    df.columns = ['Stream', 'Digit']
    df['Subset'] = subset
    dataframes.append(df)
df = pd.concat(dataframes)

# fix seed for reproducibility:
np.random.seed(0)
# extract the paths in numpy
paths = df['Stream'].to_numpy()

# normalize each path indiviually
normalized_paths = [sklearn.preprocessing.MinMaxScaler().fit_transform(path) for path in paths]
paths = normalized_paths
# define hyperparameters for the randomized Signature
def identity(x):
    return x
def sigmoid(x):
    return 1/(1+np.exp(-x))

hyperparams_dict = {
'varA':1,
'mean':0,
'res_size':10,
'activation': identity
}


# batch of hyperparas_dict['res_size']-dim feature vectors 
# print(Sigs)

# define different possible anomaly detector
clf_IF = [IsolationForest(max_samples = 100),'Isolation Forest']
clf_OCSVM = [OneClassSVM(kernel = 'rbf', nu = 0.1), 'One Class SVM']

#### change here to repeat the analysis with different classifier
clf = clf_IF
####

res_sizes = [3,7,15,31,63]
vars = [0.0001,0.005,0.001,0.05,0.1,0.3,0.5,0.7,1]
heatmap_roc = np.zeros((len(res_sizes),len(vars)))
heatmap_acc = np.zeros((len(res_sizes),len(vars)))

for i,r in enumerate(res_sizes):
    #print(i)
    for j,v in enumerate(vars):
        #print(j)
        #print('\n')
        
        hyperparams_dict['varA'] = v
        hyperparams_dict['res_size'] = r 
        # compute the randomized Signature
        Sigs = get_signature(paths,hparams = hyperparams_dict)

        mean_acc,mean_roc,_,_,_ = util.evaluate_over_digits(Sigs,df,clf[0])
        heatmap_roc[i,j] = mean_roc
        heatmap_acc[i,j] = mean_acc

plt = sns.heatmap(heatmap_roc, annot = True, fmt = '.3f', xticklabels= vars, yticklabels=res_sizes)
plt.set(xlabel = 'Variance', ylabel = 'Randomized Signature dimension', title  = 'Mean Area Under the Curve')
plt = plt.get_figure()
plt.savefig('./heatmap_roc.pdf')
plt.clf()
plt = sns.heatmap(heatmap_acc, annot = True, fmt = '.3f', xticklabels= vars, yticklabels=res_sizes)
plt.set(xlabel = 'Variance', ylabel = 'Randomized Signature dimension', title = 'Mean Accuracy')
plt = plt.get_figure()
plt.savefig('./heatmap_acc.pdf')
