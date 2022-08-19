
import sys
sys.path.append('../PenDigits/')
from RandomSignature import *
import signatory
import torch
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from tqdm.auto import tqdm

with open('../Data/SpeechCommands_Data/paths_time.pkl','rb') as f:
    paths = pickle.load(f)
with open('../Data/SpeechCommands_Data/y_train.pkl','rb') as f:
    y_train = pickle.load(f)
with open('../Data/SpeechCommands_Data/y_test.pkl','rb') as f:
    y_test = pickle.load(f)
with open('../Data/SpeechCommands_Data/y_validation.pkl','rb') as f:
    y_validation = pickle.load(f)
    
paths_torch = torch.tensor(paths)
paths_np = np.array(paths)

def sigmoid(x):
    return 1/(1+np.exp(-x))
def identity(x):
    return x

#randomized signature hyperparameters
hyperparams_dict = {
'varA':1e-06,
'mean':0,
'res_size':1463,
'activation': identity  
}
#truncation level
N=3

name = '3_id'

truncated_sig = signatory.signature(paths_torch,N).numpy()
truncated_sig_train = truncated_sig[:27864,:] 
truncated_sig_test = truncated_sig[27864:31639,:]
randomized_sig = get_signature(paths_np, vect = True, hparams = hyperparams_dict)
#train and test

# number of features of randomized signature to add
add = [100,300,400,500,700,900,1100,1300]

#fit the model with the full truncated signature
clf = RandomForestClassifier(n_estimators = 1000, max_depth = 100)
clf.fit(truncated_sig_train,y_train)
truncated_pred = clf.predict(truncated_sig_test)
accuracy_full = np.mean(truncated_pred == y_test)

clf = RandomForestClassifier(n_estimators = 1000, max_depth = 100)
mean = []
sd = []

# run N_rep simulation adding a features for each a in add
N_rep = 20
for a in tqdm(add):
    accuracy = []
    for n in range(N_rep):
        features = np.hstack([truncated_sig,randomized_sig[:,:a]])
        features_train = features[:27864,:] 
        features_test = features[27864:31639,:]
        clf.fit(features_train,y_train)
        added_pred = clf.predict(features_test)
        accuracy.append(np.mean(y_test == added_pred))
    accuracy = np.array(accuracy)
    sd.append(np.std(accuracy))
    mean.append(np.mean(accuracy))
    
    
    
with open('./mean_added_'+name+'.pkl','wb') as f:
    pickle.dump(mean,f)
with open('./sd_added_'+name+'.pkl','wb') as f:
    pickle.dump(sd,f)
    

    
