import torch
import torchaudio
from tqdm.auto import tqdm
import numpy as np
import pickle
import sklearn.preprocessing

data_path = './'

X = torch.empty(34975,16000,1)
x_ind = 0

labels_train = []
labels_test = []
labels_validation = []

#load the file in the training_list
with open(data_path+'training_list_short.txt') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        data, _ = torchaudio.load_wav(data_path+line[:-1],
                     channels_first=False,normalization = False)
        data = data/2**15       
        if len(data)!=16000:
            continue
        X[x_ind] = data
        x_ind+=1
        labels_train.append(line.split('/')[0])

        
with open(data_path+'testing_list_short.txt') as f:
    lines = f.readlines()
    for line in tqdm(lines):

        data, _ = torchaudio.load_wav(data_path+line[:-1],
                     channels_first=False,normalization = False)
        data = data/2**15
        if len(data)!=16000:
            continue
        X[x_ind] = data
        x_ind+=1
        labels_test.append(line.split('/')[0])

with open(data_path+'validation_list_short.txt') as f:
    lines = f.readlines()
    for line in tqdm(lines):

        data, _ = torchaudio.load_wav(data_path+line[:-1],
                     channels_first=False)
        data = data/2**15
        if len(data)!=16000:
            continue
        X[x_ind] = data
        x_ind+=1
        labels_validation.append(line.split('/')[0])


X = torchaudio.transforms.MFCC(log_mels=True,
                        melkwargs=dict(n_fft=100, n_mels=32),
                         n_mfcc=10)(X.squeeze(-1)).transpose(1, 2).detach()

labels = labels_train+labels_test+labels_validation
lab_enc = sklearn.preprocessing.LabelEncoder()
labels = lab_enc.fit_transform(labels)

## save the labels
with open('./y_train.pkl','wb') as f:
    pickle.dump(labels[:27864], f)

with open('./y_test.pkl','wb') as f:
    pickle.dump(labels[27864:31639], f)

with open('./y_validation.pkl','wb') as f:
    pickle.dump(labels[31639:], f)
    

## include augmentations
X_new = np.zeros((X.shape[0],X.shape[1]+1,X.shape[2]+1))
time = np.linspace(0,1,X.shape[1])
for i in range(X.shape[0]):
    # time augmentation
    data = np.hstack((time[:,None],X[i,:,:]))
    # base point augmentation
    data = np.vstack((np.zeros_like(data[0,:]),data))
    X_new[i,:,:] = data
X = X_new

## use the std scaler as done in generalised-signature-method
std_scaler = sklearn.preprocessing.StandardScaler()
std_scaler.fit(X.reshape(-1,X.shape[2]))
X_tfm = std_scaler.transform(X.reshape(-1,X.shape[2]))
X = X_tfm.reshape(X.shape)
paths_list_scaled = []

for i in tqdm(range(X.shape[0])):
    data = X[i,:,:]
    paths_list_scaled.append(data)

with open('./paths_time_basepoint.pkl','wb') as f:
    pickle.dump(X, f)
