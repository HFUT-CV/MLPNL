#include"Kmeans.h"

std::mutex mu;

Kmeans::Kmeans(cv::Mat &train_Data,cv::Mat &test_Data){

	trainData=train_Data;
	testData=test_Data;

	int testnum=sample_num-train_num;
	assert(trainData.rows==train_num*class_num);
	assert(testData.rows==testnum*class_num);
	assert(trainData.cols==testData.cols);

	int rows=testData.rows;
	int cols=testData.cols;
	attempt_count=0;

	J_min=cv::Mat::zeros(rows,2,CV_32FC1);
	for (int r = 0; r < rows; r++) {
		J_min.ptr<float>(r)[1]= 9999999.0;
	}

	for(int a=0;a<attempt_num;a++){
		std::thread t(&Kmeans::AttemptCompute,this);
		t.detach();
	}

	while(attempt_count!=attempt_num){
	}

	ResultsAnalysis();
	J_min.release();
	trainData.release();
	testData.release();
}

void Kmeans::AttemptCompute(){

	int rows=testData.rows;
	int cols=testData.cols;

	//Initialize the centroid;
	cv::Mat center;
	InitCenter(center);
	assert(center.rows==class_num&&center.cols==cols);
	
	//Temporarily store the closest distance and centroid during the iteration;
	cv::Mat J = cv::Mat::zeros(rows, 2, CV_32FC1);

	//Iteration of the distance relationship between each sample vector and the centroid;
	for(int t=0;t<T;t++){

			//The mass points are accumulated with the new position;
		cv::Mat sum_center = cv::Mat::zeros(class_num, cols, CV_32FC1);
	    cv::Mat num_center = cv::Mat::zeros(1,class_num,CV_32FC1);

		//Traverse each row of vectors;
		for(int r=0;r<rows;r++){
			float min_distance = 9999999.0;
			int tag_c = class_num + 1;

	        //Calculate the distance from individual r to each center;
			for (int c = 0; c < class_num; c++) {
				float temp = GetDistance(r, c,center);
				
				if (temp < min_distance) {
					min_distance = temp;
					tag_c = c;
				}
			}//end for c;
			J.ptr<float>(r)[0] = tag_c;
			J.ptr<float>(r)[1] = min_distance;
			
			//Prepare for the following Update Center;
			num_center.at<float>(0,tag_c)++;
			for (int j = 0; j < cols; j++) {
				sum_center.ptr<float>(tag_c)[j] += testData.ptr<float>(r)[j];
			}
		}//end for r 

		UpdateCenter(center,sum_center,num_center);

		sum_center.release();
		num_center.release();
	}

	UpdateJmin(J);

	center.release();
	J.release();
	mu.lock();
	attempt_count++;
	mu.unlock();

}

//Initialize the centroid algorithm;
void Kmeans::InitCenter(cv::Mat &Center){

	Center=cv::Mat::zeros(class_num,testData.cols,CV_32FC1);

	if(IsInitCenter==1){
		srand((unsigned)time(NULL));

		int rows=testData.rows;

		std::vector<int> visited(rows, 0);

		for (int i = 0; i < class_num; i++) {
			int temp = (int)rand() % rows;
			while (visited[temp] == 1)
			{
				temp = (int)rand() % rows;
			}
			visited[temp] = 1;

			testData.row(temp).copyTo(Center.row(i));
		}
		visited.clear();
	}else{

		for(int i=0;i<class_num;i++){

			cv::Mat sum_center=cv::Mat::zeros(1,trainData.cols,CV_32FC1);

			for(int j=0;j<train_num;j++){

				int tag=i*train_num+j;
				sum_center+=trainData.row(tag);
			}
			sum_center=sum_center/train_num;
			sum_center.copyTo(Center.row(i));
		}
	}
}

inline float Kmeans::GetDistance(int r, int c,cv::Mat &center){
	
	float temp = 0;

	cv::Mat temp1,temp2;

	temp1=testData.row(r);
	temp2=center.row(c);

	cv::Mat temp3=temp1-temp2;

	temp3=temp3*temp3.t();
	temp=temp3.ptr<float>(0)[0];
	temp = sqrt(temp);  //Adjust the distance formula.
	return temp;
}

inline void Kmeans::UpdateCenter(cv::Mat &center,cv::Mat &sum_center,cv::Mat &num_center){

	static int mmm = 0;
	for (int i = 0; i < class_num; i++) {
		for (int j = 0; j < testData.cols; j++) {
			center.ptr<float>(i)[j] = sum_center.ptr<float>(i)[j] / num_center.ptr<float>(0)[i];
		}
	}

	mmm++;
	std::cout << "update:" << mmm << std::endl;
}

void Kmeans::UpdateJmin(cv::Mat J){
	mu.lock();
	for (int r = 0; r < testData.rows; r++) {
		if (J_min.ptr<float>(r)[1] > J.ptr<float>(r)[1]) {
			J_min.ptr<float>(r)[0] = J.ptr<float>(r)[0];
			J_min.ptr<float>(r)[1] = J.ptr<float>(r)[1];
		}
	}//end for 
	mu.unlock();
}

void Kmeans::ResultsAnalysis(){

	int rows=testData.rows;
	int cols=testData.cols;
	int testnum=sample_num-train_num;

	assert(J_min.rows=rows);

	int *table = new int[class_num];

	float class_true_num = 0;

	float sum_true_num = 0;

	std::ofstream fout("res.txt");
	fout << J_min << std::endl;

	for (int i = 0; i<class_num; i++) {

		//Refresh table
		for (int k = 0; k<class_num; k++) {
			table[k] = 0;
		}

		for (int j = 0; j<testnum; j++) {
			table[(int)J_min.ptr<float>(i*testnum + j)[0]]++;
		}

		std::sort(table, table + class_num);   //Call the sort function to sort to get the largest cluster.

		class_true_num = table[class_num - 1];

		sum_true_num += table[class_num - 1];

		double class_rate = class_true_num / testnum;

		fout << i << "th " << "Recognition rate" << class_rate << std::endl;
	}

	double sum_rate = sum_true_num / (class_num*testnum);

	
	fout << sum_rate << std::endl;   //Save the recognition rate under the res.txt file.
	fout.close();

}
