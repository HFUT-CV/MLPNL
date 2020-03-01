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

	int block_stride;        //block�Ļ������ڵĲ�����Ĭ�����ó�block��һ�볤�ȣ�

	cv::Mat src_img;          //�����ԴͼƬ��

	std::vector<cv::Mat> train_images;

	int D1;

	int D2;

	cv::Size Block_num;            //һ��ͼƬ�ֳɵ�DFD��������

	cv::Size Block_Size;           //ÿ��DFD����ĳߴ��С��

	int perImg_PDM_Num;            //��ʾÿ��ͼƬ��������PDM����

	std::vector<double> DFD_vec;   //���õ���DFD������

								   //int featureDim;                //ÿ����Ƭ�õ�DFD_vecά�ȣ�

	std::vector<cv::Mat> W;

	std::vector<cv::Mat> V;

	std::vector<cv::Mat>W1;
	std::vector<cv::Mat>V1;

	std::vector<cv::Mat>W2;
	std::vector<cv::Mat>V2;

	std::vector<cv::Mat>W3;
	std::vector<cv::Mat>V3;



	void myNet(cv::Mat A);

	void Extract_PDM();     //��һ��ͼƬimage������ȡPDM����

	cv::Mat Create_PDM(int i, int j);         //����һ����Ӧ��PDM����

	void DFD_Train();                    //����DFD����ѵ����

	void _2D_LDA(cv::Mat A);         //2D_LDA�㷨���е�����
	std::vector<cv::Mat> _2D_LDA(std::vector<cv::Mat>A,int w_cols,int v_cols,int tag);         //2D_LDA�㷨���е�����
	void _2D_LDA_new(cv::Mat A);

	void Compute_DFDVec();               //����ѵ���õ����ݣ���������DFD��������DFD_vec
	void Compute_DFDVec_new();

	void knn_classification();   //KNNѵ�������з�����㣻

	void SVM_classification();

	void Kmeans_classification();
	void LR_classification();

	cv::Mat NormalizeVec(cv::Mat Vec);    //���������Ĺ�һ����

	cv::Mat Extract_PDM(cv::Mat img);    //��һ��ͼƬ������ȡPDM����
	cv::Mat Extract_PDM_new(cv::Mat img);

	cv::Mat Mat_Map_HashBinary(cv::Mat img);   //��Mat����ӳ���hash�����ƾ���

	cv::Mat sign_B(cv::Mat A);  //����ͨ����ӳ��ɶ����ƾ���

	cv::Mat Compute_Mean_of_Matrix(cv::Mat A);

	//���ֲ�ͬ�ؼ������
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

	void Process();             //��������Ԥ������������
};