#include"myKmeans.h"
#include<thread>
#include<mutex>

std::mutex mu;

myKmeans::myKmeans(cv::Mat &Data, cv::Mat &Best_label, int K, int Attempts, int Iterations_num) {

	attempt_count = 0;

	data = Data;

	best_label = Best_label;

	k = K;

	attempts = Attempts;

	iterations_num = Iterations_num;

	//	criteria_epsilon = Criteria_epsilon;

	Is_return = false;

	Thread_Compute();
}

myKmeans::~myKmeans() {

	data.release();

	best_label.release();

	J_min.release();

}


void myKmeans::Thread_Compute() {

	clock_t start, finish;
	start = clock();

	assert(k != 0);

	assert(data.rows == best_label.rows);

	assert(data.cols != 0);

	assert(data.rows >= k);

	rows = data.rows;
	cols = data.cols;

	J_min = cv::Mat::zeros(rows, 2, CV_32FC1);

	for (int r = 0; r < rows; r++) {
		J_min.ptr<float>(r)[1] = 9999999.0;
	}


	//因为attempts模块是相互独立的，所以可以采用多线程一起跑；
	for (int i = 0; i<attempts; i++) {
		std::thread t(&myKmeans::Attempt_Compute, this);
		t.detach();
	}

	//因为采用的是主线程和分线程相互隔离的方法，要保证attempts模块全部跑完，才能继续往下跑，这里写个死循环等待；
	while (attempt_count < attempts) {
		
		//std::cout << attempt_count << std::endl;
	}

//	std::thread t_temp(&myKmeans::temp_thread, this);
//	t_temp.join();



	//	std::thread t1(&myKmeans::Attempt_Compute,this);
	//	std::thread t2(&myKmeans::Attempt_Compute, this);
	//	std::thread t3(&myKmeans::Attempt_Compute, this);
	//	t1.join();
	//	t2.join();
	//	t3.join();


	J_min.col(0).copyTo(best_label.col(0));


	finish = clock();

	double run_time = (double)(finish - start) / CLOCKS_PER_SEC;
	std::cout << run_time << std::endl;

	return;
}

void myKmeans::Attempt_Compute() {

	//初始化质点；
	cv::Mat _center = cv::Mat::zeros(k, cols, CV_32FC1);

	//质点跟新位置累计；
	cv::Mat _sum_center = cv::Mat::zeros(k, cols, CV_32FC1);

	int *_num_center = new int[k];
	for (int n = 0; n < k; n++) {
		_num_center[n] = 0;
	}

	//迭代过程中暂时性存放最近距离和质点；
	cv::Mat _J = cv::Mat::zeros(rows, 2, CV_32FC1);

	InitCenter(_center);

	assert(_center.rows == k);
	assert(_center.cols == cols);


	for (int t = 0; t < iterations_num; t++) {

		//遍历每一个个体，确定与之最近的质心；

		for (int r = 0; r < rows; r++) {

			float min_distance = 9999999.0;
			int tag_c = k + 1;

			//计算个体 r 到每一个center的距离；
			for (int c = 0; c < k; c++) {
				float temp = GetDistance(r, c, _center);

				if (temp < min_distance) {
					min_distance = temp;
					tag_c = c;
				}

			}//end for c;

			_J.ptr<float>(r)[0] = tag_c;
			_J.ptr<float>(r)[1] = min_distance;

			//为后面的Update Center 做准备;
			_num_center[tag_c]++;
			for (int j = 0; j < cols; j++) {
				_sum_center.ptr<float>(tag_c)[j] += data.ptr<float>(r)[j];

			}


		}//end for r

		UpdateCenter(_center, _sum_center, _num_center);

	}//end for t;

	J_min_Update(_J);

	delete _num_center;
	_center.release();
	_sum_center.release();
	_J.release();

	mu.lock();
	attempt_count++;
	mu.unlock();
}


inline void myKmeans::UpdateCenter(cv::Mat &center, cv::Mat &sum_center, int *num_center) {

	//	old_center=center;

	static int mmm = 0;
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < data.cols; j++) {
			center.ptr<float>(i)[j] = sum_center.ptr<float>(i)[j] / num_center[i];
		}
	}

	mmm++;
	std::cout << "update:" << mmm << std::endl;

}

void myKmeans::InitCenter(cv::Mat &center) {

	srand((unsigned)time(NULL));

	std::vector<int> visited(data.rows, 0);

	for (int i = 0; i < k; i++) {
		int temp = (int)rand() % data.rows;

		while (visited[temp] == 1)
		{
			temp = (int)rand() % data.rows;
		}

		visited[temp] = 1;

		float *data_begin, *data_end, *data_ptr;

		data_begin = (float*)data.data + temp*data.cols;
		data_end = (float*)data_begin + data.cols;
		data_ptr = (float *)center.data + i*data.cols;
		std::copy(data_begin, data_end, data_ptr);

	}

	visited.clear();
}

void myKmeans::J_min_Update(cv::Mat &J) {
	mu.lock();

	for (int r = 0; r < rows; r++) {

		if (J_min.ptr<float>(r)[1] > J.ptr<float>(r)[1]) {

			J_min.ptr<float>(r)[0] = J.ptr<float>(r)[0];
			J_min.ptr<float>(r)[1] = J.ptr<float>(r)[1];

		}
	}//end for 

	mu.unlock();
}

inline float myKmeans::GetDistance(int r, int c, cv::Mat &center) {

	float temp = 0;

/*	for (int i = 0; i < data.cols; i++) {
		float xx = data.ptr<float>(r)[i] - center.ptr<float>(c)[i];
		temp += xx*xx;
		//		temp = sqrt(temp);
	}
*/
	cv::Mat temp1=cv::Mat::zeros(1,data.cols,CV_32FC1);
	cv::Mat temp2=cv::Mat::zeros(1,data.cols,CV_32FC1);

	//	data.row(r).copyTo(temp1);
	//	data.row(c).copyTo(temp2);

	temp1=data.row(r);
	temp2=center.row(c);

	cv::Mat temp3=temp1-temp2;

	temp3=temp3*temp3.t();
	temp=temp3.ptr<float>(0)[0];

	return temp;
}

/*
inline void myKmeans::J_Update(int r,cv::Mat &_center,cv::Mat &_J,cv::Mat &_sum_center,int *_num_center,int &row_count){

float min_distance = 9999999.0;
int tag_c = k + 1;

//计算个体 r 到每一个center的距离；
for (int c = 0; c < k; c++) {
float temp = GetDistance(r, c,_center);

if (temp < min_distance) {
min_distance = temp;
tag_c = c;
}
}//end for c;

_J.ptr<float>(r)[0] = tag_c;
_J.ptr<float>(r)[1] = min_distance;

//为后面的Update Center 做准备;
_num_center[tag_c]++;
for (int j = 0; j < cols; j++) {
_sum_center.ptr<float>(tag_c)[j] += data.ptr<float>(r)[j];
}

mu.lock();
row_count++;
mu.unlock();
}
*/
