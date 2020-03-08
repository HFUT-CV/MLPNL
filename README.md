# MLPNL
A Multilayer Pyramid Network Based on Learning for Vehicle Logo Recognition

## How to Use
The code for MLPNL includes two part : training and inference. The API of training and inference is writed in [myLBVLD.cpp](https://github.com/HFUT-VL/MLPNL/blob/master/code/myLBVLD.cpp). You can get the API in the function called **myLBVLD::Process**. You should install OpenCV 4.0, cuda 9, cudnn 7 or above before you use the code.

#### Parameters
In the head of myLBVLD.cpp, you can set some parameters, such as block size, cell size and so on. However, if you want to get best performance, you maybe need to read the paper of MLPNL. The paper will tell how to set and adjust the parameters.

#### Training 
When you would like to training your dataset, you maybe need these function : Extract_PDM(),DFD_Train() and cluster() and open ontes. These functions will learning the feature parameters from the your dataset. 

#### Inference
when you use the api of inference, you must make sure that the feature parameter files already exist. Now, you need the function of  Compute_DFDVec_KMeans and classifier function such as knn_classification() or SVM_classification().
