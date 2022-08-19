import sys
sys.path.append('../PenDigits/')
from RandomSignature import *
import signatory
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
    
paths_torch = torch.tensor(paths)
paths_np = np.array(paths)

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
truncated_sig = signatory.signature(paths_torch,3).numpy()
truncated_sig_train = truncated_sig[:27864,:] 
truncated_sig_test = truncated_sig[27864:31639,:]
randomized_sig = get_signature(paths_np, vect = True, hparams = hyperparams_dict)



rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
N_rep = 20

## fit the model using the truncated signature only
clf = RandomForestClassifier(n_estimators = 1000, max_depth = 100)
clf.fit(truncated_sig_train,y_train)
truncated_pred = clf.predict(truncated_sig_test)
accuracy_full = np.mean(truncated_pred == y_test)

sd = []
mean = []

for r in rates:
    accuracy = []
    for i in range(N_rep):
    
        truncated_sig_rate = r
        ind = np.random.choice(range(1463),replace = False,size = int(1463*truncated_sig_rate))


        truncated_sig_subsampled = truncated_sig[:,ind]
            
        number_rsig = 1463 - int(1463*(truncated_sig_rate))
        randomized_sig_subsampled = randomized_sig[:,:number_rsig]
        
        #merge the rand sig and trun sig features
        features = np.hstack([truncated_sig_subsampled,randomized_sig_subsampled])
        
        ## train and evaluate
        features_train = features[:27864,:] 
        features_test = features[27864:31639,:]

        # train using the features 
        clf = RandomForestClassifier(n_estimators = 1000, max_depth = 100)
        clf.fit(features_train,y_train)
        merged_pred = clf.predict(features_test)
        accuracy.append(np.mean(y_test == merged_pred))
    
    accuracy = np.array(accuracy)
    sd.append(np.std(accuracy))
    mean.append(np.mean(accuracy))


#save the results
with open('./full_accuracy_merged_1463_id.pkl','wb') as f:
    pickle.dump(accuracy_full,f)

with open('./mean_merged_1463_id.pkl','wb') as f:
    pickle.dump(mean,f)

with open('./sd_merged_1463_id.pkl','wb') as f:
   pickle.dump(sd,f)
