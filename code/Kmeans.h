#pragma once

#include<iostream>
#include<fstream>
#include<opencv2\opencv.hpp>
#include<opencv2\core\core.hpp>
#include<ctime>
#include<thread>
#include<mutex>

const int train_num = 5;                                          //ÿ��������ѵ������

const int sample_num = 200;                                       //ÿ����������������

const int class_num = 80;                                         //����������;

const int IsInitCenter=0;       //���ĳ�ʼ����ʽ��0Ϊ��ѵ�������������ģ�1Ϊ�ڲ�������������������ģ�

const int attempt_num=4;        //�ظ���ʼ���Ĵ�����

const int T=100;   //����������

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