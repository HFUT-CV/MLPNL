#pragma once

#include<iostream>
#include<fstream>
#include<opencv2\opencv.hpp>
#include<opencv2\core\core.hpp>
#include<ctime>
#include<thread>
#include<mutex>

const int train_num = 5;                                          //Training number of each class;

const int sample_num = 200;                                       //The number of samples in each class;

const int class_num = 80;                                         //Number of classes in the sample;

const int IsInitCenter=0;       //The centroid initialization method, 0 is used to generate centroids from training samples, and 1 is used to randomly generate centroids in test samples;

const int attempt_num=4;        //The number of repeated initializations;

const int T=100;   //Number of iterations

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
