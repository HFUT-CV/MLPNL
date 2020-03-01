#pragma once

#include<opencv2\core\core.hpp>
#include<opencv2\opencv.hpp>
#include<opencv2\imgproc\imgproc.hpp>

#include<iostream>
#include<fstream>
#include<ctime>
#include<vector>
#include<stdlib.h>


/*
���㷨 ������������������С��ȷ��˫����,
��ʼ���Ĳ���������ɣ�
*/


class myKmeans {

public:

	myKmeans(cv::Mat &Data, cv::Mat &Best_label, int K, int Attempts, int Iterations_num);
	~myKmeans();

	cv::Mat best_label;             //���������ı�ǩ��������Ӧ����data��������

private:

	bool Is_return;                //��ֵ��ʼ��Ϊfalse,����ٸ����ʵ��ʱ��ǰ���������Ƴ�kmeans.

	cv::Mat data;                    //����������ݣ�mat�����ÿһ��Ϊһ�����塣

	int k;                          //������������Ŀ��

	int attempts;                    //Kmeans �㱻ִ�еĴ�����

	int iterations_num;              //������������

	//	double criteria_epsilon;          //���ľ�ȷ�ȣ�Ҫ�ǵ��������ȷ�ȣ���õ�ı�ǩ�����̶������ٲ��������

	cv::Mat J_min;                      //����ÿ������ı�ǩ�����µ����ľ��룻


	int rows;

	int cols;

	volatile int attempt_count;     //volatile�ؼ��ֵ�ʹ�ã�����������Ķ��̵߳��Զ��Ż������������ڼĴ������У����޷���ȡ�������޸ġ�volatile����ÿ�ζ�ȡ�����ڴ浱�ж�ȡ


	//���̺߳�������;

	void Thread_Compute();

	void Attempt_Compute();

	void InitCenter(cv::Mat &Center);

	inline void UpdateCenter(cv::Mat &center, cv::Mat &sum_center, int *num_center);

	void J_min_Update(cv::Mat &J);

	inline float GetDistance(int r, int c, cv::Mat &center);

	//	inline void J_Update(int r,cv::Mat &center,cv::Mat &J,cv::Mat &_sum_center,int *_num_center,int &row_count);

};


