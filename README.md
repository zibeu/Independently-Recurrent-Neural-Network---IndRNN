# Independently-Recurrent-Neural-Network---IndRNN
Independently Recurrent Neural Network - IndRNN using "NTU RGB+D" Action Recognition Dataset


Steps:
    Place Skeleton data folder "nturgb+d_skeletons"
    Createa a folder for numpy files "nturgb_npy"
    Remove missing data files: ref: https://github.com/shahroudy/NTURGB-D"
    Run "skeleton_to_numpy.py"
    Once Numpy files are generated, adjust file "opts" for hyperparameter setting
    Run "Indrnn_action_train.py"
    
 For more information on the model read: https://github.com/Sunnydreamrain/IndRNN_pytorch/blob/master/Independently%20Recurrent%20Neural%20Network%20(IndRNN)%20Building%20A%20Longer%20and%20Deeper%20RNN.pdf
