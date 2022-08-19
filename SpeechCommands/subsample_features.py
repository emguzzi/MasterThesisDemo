import sys
sys.path.append('../PenDigits/')
from RandomSignature import *
import torch
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

with open('../Data/SpeechCommands_Data/paths_time.pkl','rb') as f:
    paths = pickle.load(f)
with open('../Data/SpeechCommands_Data/y_train.pkl','rb') as f:
    y_train = pickle.load(f)
with open('../Data/SpeechCommands_Data/y_test.pkl','rb') as f:
    y_test = pickle.load(f)
with open('../Data/SpeechCommands_Data/y_validation.pkl','rb') as f:
    y_validation = pickle.load(f)
    
paths_np = np.array(paths)
paths_torch = torch.tensor(paths)

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def identity(x):
    return x

hyperparams_dict = {
'varA':1e-06,
'mean':0,
'res_size':1463,
'activation': identity  
}
sig = get_signature(paths_np, vect = True, hparams = hyperparams_dict)

## uncomment this to use the truncated signature 
# sig = signatory.signature(paths_torch,3)


#train and test
sig_train = sig[:27864,:] 
sig_test = sig[27864:31639,:]


## 
rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]

N_rep = 20


#fit the model using the complete features set
clf = RandomForestClassifier(n_estimators = 1000, max_depth = 100)
clf.fit(sig_train,y_train)
pred_full = clf.predict(sig_test)
accuracy_full = np.mean(pred_full == y_test)

sd = []
mean = []
for r in rates:
    accuracy = []
    for i in range(N_rep):
    
        #subsample the features
        omit_rate = r
        ## for the truncation sig change 1463 to 132 if considering the truncation level N=2
        ## for the randomized sig change 1463 to 132 if the dim in the hyperparams_dict is changed
        ind = np.random.choice(range(1463),replace = False,size = int(1463*(1-omit_rate)))
        sig_train_subsampled = sig[:27864,ind] 
        sig_test_subsampled = sig[27864:31639,ind]

        # fit the model on the subsampled fetures set 
        clf.fit(sig_train_subsampled,y_train)
        pred_subsampled = clf.predict(sig_test_subsampled)
        accuracy.append(np.mean(y_test == pred_subsampled))
        
    accuracy = np.array(accuracy)
    sd.append(np.std(accuracy))
    mean.append(np.mean(accuracy))


#save the results
with open('./full_accuracy_1463_randomized_id.pkl','wb') as f:
    pickle.dump(accuracy_full,f)
with open('./mean_1463_randomized_id.pkl','wb') as f:
    pickle.dump(mean,f)
with open('./sd_1463_randomized_id.pkl','wb') as f:
   pickle.dump(sd,f)
