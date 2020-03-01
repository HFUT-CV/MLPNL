#pragma once

#include<iostream>
#include<fstream>
#include<opencv2\opencv.hpp>
#include<opencv2\core\core.hpp>
#include<ctime>
#include<thread>
#include<mutex>

const int train_num = 5;                                          //每类样本的训练数；

const int sample_num = 200;                                       //每类样本的样本数；

const int class_num = 80;                                         //样本的类数;

const int IsInitCenter=0;       //质心初始化方式，0为用训练样本生成质心，1为在测试样本中随机生成质心；

const int attempt_num=4;        //重复初始化的次数；

const int T=100;   //迭代次数；

class Kmeans{

private:

	cv::Mat trainData;
	cv::Mat testData;

	cv::Mat J_min;  
	volatile int attempt_count;

	void InitCenter(cv::Mat &Center);
	void AttemptCompute();
	inline float GetDistance(int r, int c,cv::Mat &center);
	inline void UpdateCenter(cv::Mat &center,cv::Mat &sum_center,cv::Mat &num_center);
	void UpdateJmin(cv::Mat J);
	void ResultsAnalysis();
	
public:
	Kmeans(cv::Mat &train_Data,cv::Mat &test_Data);
};