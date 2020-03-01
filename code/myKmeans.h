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
本算法 采用最大迭代次数和最小精确度双控制,
初始质心采用随机生成；
*/


class myKmeans {

public:

	myKmeans(cv::Mat &Data, cv::Mat &Best_label, int K, int Attempts, int Iterations_num);
	~myKmeans();

	cv::Mat best_label;             //待聚类个体的标签，其行数应等于data的行数。

private:

	bool Is_return;                //该值初始化为false,如果再更新质点的时候，前后相差不大，则推出kmeans.

	cv::Mat data;                    //待聚类的数据，mat矩阵的每一行为一个个体。

	int k;                          //被分类的类别数目；

	int attempts;                    //Kmeans 算被执行的次数；

	int iterations_num;              //最大迭代次数；

	//	double criteria_epsilon;          //质心精确度，要是低于这个精确度，这该点的标签将被固定，不再参与迭代；

	cv::Mat J_min;                      //保存每个个体的标签和最下的质心距离；


	int rows;

	int cols;

	volatile int attempt_count;     //volatile关键字的使用，避免编译器的多线程的自动优化，将变量至于寄存器当中，而无法读取变量的修改。volatile限制每次读取都从内存当中读取


	//多线程函数部分;

	void Thread_Compute();

	void Attempt_Compute();

	void InitCenter(cv::Mat &Center);

	inline void UpdateCenter(cv::Mat &center, cv::Mat &sum_center, int *num_center);

	void J_min_Update(cv::Mat &J);

	inline float GetDistance(int r, int c, cv::Mat &center);

	//	inline void J_Update(int r,cv::Mat &center,cv::Mat &J,cv::Mat &_sum_center,int *_num_center,int &row_count);

};


