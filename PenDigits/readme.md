# Pen Digits dataset

### PenDigitDemo
This notebook allows to inspect the dataset and try out different model configurations. It is also possible to visualize the randomized signature of 
dimension $k=3$ and the cumulative distribution plots for varying value of $k$. The notebook can be used to reproduce the results in Table 1 and Table 3
of the thesis as well as to obtain Figure 4.

### parameters_heatmap
Contains the code needed to explore the dependency of the model on the hyperparameters and to recreate the heatmaps in Figure 5 of the thesis.

### RandomSignature
Contains the functions needed to compute the randomized signature. Also in [SKAB](https://github.com/emguzzi/MasterThesisDemo/tree/main/SKAB) and 
[SpeechCommands](https://github.com/emguzzi/MasterThesisDemo/tree/main/SpeechCommands) the computations of the randomized signature will rely on these funcitons.

### util
Contains the function to load the dataset and the functions used to evaluate the model in the PenDigitDemo notebook. 
