# Speech Commands dataset

### SpeechCommands_Demo
The notebook can be used to inspect the model and try different hyperparameters both for the randomized and truncated signature. To run the code one need to upload the preprocessed data in google drive.

### read_preprocess
The file can be used to read the .wav file of the dataset and to consider the needed preprocessing steps. We only consider a subset of the recordings specified in the files train\_list\_short.txt, test\_list\_short.txt and validation\_list\_short.txt in the [SpeechCommands_Data](https://github.com/emguzzi/MasterThesisDemo/tree/main/Data/SpeechCommands_Data)

### subsample_features
The file can be used to experiment how the accuracy changes when we subsample the randomized and truncated signature. It can be used to recreate the plots in Figure 9 and Figure 10.

### add_features
The file can be used to experiment how the accuracy changes when we add a part of the randomized signature to the truncated signature. It can be used to recreate the plots in Figure 11.

### merge_features
The file can be used to experiment how the accuracy changes when we consider a mixture of features coming both from the randomized and truncated signature. It can be used to recreate the plost in Figure 12.
