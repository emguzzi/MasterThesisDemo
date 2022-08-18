## grid search for best hyperparameters for the SKAB dataset
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import itertools
import utils
import sys
sys.path.append('../PenDigits')
from RandomSignature import *

## hyperparameters
def identity(x):
    return x
def sigmoid(x):
    return 1/(1+np.exp(-x))

hyperparams_dict = {
'varA':1,
'mean':0,
'res_size':100,
'activation': sigmoid
}
    
## define grid
vars = [0.00001,0.0001,0.001,0.01,0.1,1,10]
dimensions = [100,500]
activations = [sigmoid,identity]
subs = [3,5,10,20,30]
conts = [0.05]
clfs = ['SVM']#,'IF']

## define grid
vars = [0.1,1,10]
dimensions = [100,500]
activations = [identity]
subs = [30]
conts = [0.05]
clfs = ['SVM']#,'IF']

# set name that identifies the file with results
name = 'SVM_test'

parameters = itertools.product(vars,dimensions,activations,subs,conts,clfs)
list_results = []
for p in tqdm(parameters):
    hyperparams_dict['varA'] = p[0]
    hyperparams_dict['res_size'] = p[1]
    hyperparams_dict['activation'] = p[2]
    F1, FAR, MAR = utils.evaluate(hyperparams_dict,p[3],p[5],p[4])
    list_results += [p[0],p[1],p[2],p[3],p[4],p[5],F1, FAR, MAR]
    with open('results.txt','a') as f:
        f.write(str(''.join(str(x)+' , ' for x in [p[0],p[1],p[2],p[3],p[4],p[5],F1, FAR, MAR])))
        f.write('\n')
results = pd.DataFrame(data = np.reshape(np.array(list_results),(-1,11)),columns = ['varA','res_size','activation','len_sub','contamination','clf','F1', 'FAR', 'MAR'])

results.sort_values(by = ['F1','FAR','MAR'])
results.to_csv('results_'+name+'.csv')
