# MLPNL
A Multilayer Pyramid Network Based on Learning for Vehicle Logo Recognition

## How to Use
The code for MLPNL includes two parts: training and inference. The API of training and inference is writen in [myLBVLD.cpp](https://github.com/HFUT-VL/MLPNL/blob/master/code/myLBVLD.cpp). You can get the API in the function called **myLBVLD::Process()**. You should install OpenCV 4.0, cuda 9, cudnn 7 or above before you use the code.

#### Parameters
In the head of myLBVLD.cpp, you can set some parameters, such as block size, cell size and so on. However, if you want to get best performance, you need to read the paper of MLPNL. The paper can guide you how to set and adjust the parameters.

#### Training 
If you plan to train your own dataset, you need these function: Extract_PDM(),DFD_Train() and cluster() and code comments. These functions will learn the feature parameters from  your dataset. 

#### Inference
When you use the API of inference, you must make sure that the feature parameter files already exist. Now, you need the function of  Compute_DFDVec_KMeans() and classifier function such as knn_classification() or SVM_classification().
