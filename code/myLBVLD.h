#pragma once

#include<iostream>
#include<string>
#include<fstream>

#include<opencv2\core\core.hpp>
#include<opencv2\opencv.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\ml\ml.hpp>
#include<opencv2\highgui\highgui.hpp>

#include<ml\ml.hpp>
#include"Kmeans.h"

class myLBVLD {
private:

	int block_stride;        //block的滑动窗口的步长，默认设置成block的一半长度；

	cv::Mat src_img;          //读入的源图片；

	std::vector<cv::Mat> train_images;

	int D1;

	int D2;

	cv::Size Block_num;            //一张图片分成的DFD区域数；

	cv::Size Block_Size;           //每个DFD区域的尺寸大小；

	int perImg_PDM_Num;            //表示每张图片所产生的PDM数；

	std::vector<double> DFD_vec;   //最后得到的DFD特征；

								   //int featureDim;                //每张照片得到DFD_vec维度；

	std::vector<cv::Mat> W;

	std::vector<cv::Mat> V;

	std::vector<cv::Mat>W1;
	std::vector<cv::Mat>V1;

	std::vector<cv::Mat>W2;
	std::vector<cv::Mat>V2;

	std::vector<cv::Mat>W3;
	std::vector<cv::Mat>V3;



	void myNet(cv::Mat A);

	void Extract_PDM();     //从一张图片image当中提取PDM矩阵；

	cv::Mat Create_PDM(int i, int j);         //生成一个对应的PDM矩阵；

	void DFD_Train();                    //进行DFD特征训练；

	void _2D_LDA(cv::Mat A);         //2D_LDA算法进行迭代；
	std::vector<cv::Mat> _2D_LDA(std::vector<cv::Mat>A,int w_cols,int v_cols,int tag);         //2D_LDA算法进行迭代；
	void _2D_LDA_new(cv::Mat A);

	void Compute_DFDVec();               //根据训练好的数据，我们生成DFD特征向量DFD_vec
	void Compute_DFDVec_new();

	void knn_classification();   //KNN训练器进行分类计算；

	void SVM_classification();

	void Kmeans_classification();
	void LR_classification();

	cv::Mat NormalizeVec(cv::Mat Vec);    //特征向量的归一化；

	cv::Mat Extract_PDM(cv::Mat img);    //从一张图片当中提取PDM矩阵
	cv::Mat Extract_PDM_new(cv::Mat img);

	cv::Mat Mat_Map_HashBinary(cv::Mat img);   //将Mat矩阵映射成hash二进制矩阵；

	cv::Mat sign_B(cv::Mat A);  //将普通矩阵映射成二进制矩阵；

	cv::Mat Compute_Mean_of_Matrix(cv::Mat A);

	//三种不同地激活函数；
	float Sign_Activation(float i);
	float Sigmod_Activation(float i);
	float tanH_Activation(float i);

	std::vector<cv::Mat> learn_w(std::vector<cv::Mat>A, cv::Mat  &w, cv::Mat v);
	std::vector<cv::Mat> learn_v(std::vector<cv::Mat>A, cv::Mat  w, cv::Mat &v);
	std::vector<cv::Mat> learn_w_and_v(std::vector<cv::Mat>A, cv::Mat  &w, cv::Mat &v);
	void CreateNet(cv::Mat A);
	float Createloss(std::vector<cv::Mat>A, cv::Mat  w, cv::Mat v);
	float loss_2(std::vector<cv::Mat>A, cv::Mat  w, cv::Mat v);

	cv::Mat codebook1;
	cv::Mat codebook2;
	cv::Mat codebook3;
	void cluster();
	void Compute_DFDVec_KMeans();
	cv::Mat extractPDM_KMeans(cv::Mat img);
	int compute_Dis(cv::Mat A,cv::Mat codebook);

	void duoxianchenglala(int i);



public:
	myLBVLD();

	void Process();             //样本数据预处理等相关内容
};