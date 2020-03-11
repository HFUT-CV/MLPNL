#include"myLBVLD.h"
#include<fstream>
#include<thread>

//Parameter adjustment area:

const cv::Size img_size = cv::Size(62, 62);                      //Size of Image 

const int IsOverLapping = 1;                                     //Whether to use overlap when extracting PDMs : 0  no overlap, 1  overlap.

const std::string filepath = "D:\\data80\\";                     //File path

const int trainnum = 10;                                          //Training number of each class

const int samplenum = 200;                                       //Total number of samples in each class

const int classnum = 78;                                         //Number of classes



const int block_r = 3;                                             //Radius of image block

const int r = 3;                                                   //Sampling radius

																   //const int table_r[24][2]={{-2,-2},{-2,-1},{-2,0},{-2,1},{-2,2},{-1,2},{0,2},{1,2},
																   //	                     {2,2},{2,1},{2,0},{2,-1},{2,-2},{1,-2},{0,-2},{-1,-2},
																   //	                     {-1,-1},{-1,0},{-1,1},{0,1},{1,1},{1,0},{1,-1},{0,-1}};


const int table_r[48][2] = { { -3,-3 },{ -3,-2 },{ -3,-1 },{ -3,0 },{ -3,1 },{ -3,2 },{ -3,3 },{ -2,3 },{ -1,3 },{ 0,3 },{ 1,3 },{ 2,3 },{ 3,3 },
{ 3,2 },{ 3,1 },{ 3,0 },{ 3,-1 },{ 3,-2 },{ 3,-3 },{ 2,-3 },{ 1,-3 },{ 0,-3 },{ -1,-3 },{ -2,-3 },{ -2,-2 },{ -2,-1 },
{ -2,0 },{ -2,1 },{ -2,2 },{ -1,2 },{ 0,2 },{ 1,2 },{ 2,2 },{ 2,1 },{ 2,0 },{ 2,-1 },{ 2,-2 },{ 1,-2 },{ 0,-2 },{ -1,-2 },
{ -1,-1 },{ -1,0 },{ -1,1 },{ 0,1 },{ 1,1 },{ 1,0 },{ 1,-1 },{ 0,-1 } };
/*
const int table_r[80][2] = { {-4,-4},{-4,-3},{-4,-2},{-4,-1},{-4,0},{-4,1},{-4,2},{-4,3},{-4,4},{-3,4},{-2,4},{-1,4},{0,4},{1,4},{2,4},{3,4},{4,4},
{4,3},{4,2},{4,1},{4,0},{4,-1},{4,-2},{4,-3},{4,-4},{3,-4},{2,-4},{1,-4},{0,-4},{-1,-4},{-2,-4},{-3,-4},{ -3,-3 },{ -3,-2 },{ -3,-1 },{ -3,0 },{ -3,1 },{ -3,2 },{ -3,3 },{ -2,3 },{ -1,3 },{ 0,3 },{ 1,3 },{ 2,3 },{ 3,3 },
{ 3,2 },{ 3,1 },{ 3,0 },{ 3,-1 },{ 3,-2 },{ 3,-3 },{ 2,-3 },{ 1,-3 },{ 0,-3 },{ -1,-3 },{ -2,-3 },{ -2,-2 },{ -2,-1 },
{ -2,0 },{ -2,1 },{ -2,2 },{ -1,2 },{ 0,2 },{ 1,2 },{ 2,2 },{ 2,1 },{ 2,0 },{ 2,-1 },{ 2,-2 },{ 1,-2 },{ 0,-2 },{ -1,-2 },
{ -1,-1 },{ -1,0 },{ -1,1 },{ 0,1 },{ 1,1 },{ 1,0 },{ 1,-1 },{ 0,-1 } };*/


/*const int table_r[24][2] = { {-3,-3},{-3,0},{-3,3},{-2,2},{-2,0},{-2,2},{-1,1},{-1,0},{-1,1},{0,-3},{0,-2},{0,-1},{0,1},{0,2},{0,3},{1,-1},{1,0},{1,1},{2,-2},
{2,0},{2,2},{3,-3},{3,0},{3,3} };
*/

const float sita_w = 1;
const float sita_b = 2.5;
const float sita_u = 1;  //The difference in activation function, the smaller the better
const float sita_s = 1.5;  //The divergence of self distribution, the bigger the better

//const int table_r[8][2]={{-1,-1},{-1,0},{-1,1},{0,1},{1,1},{1,0},{1,-1},{0,-1}};

myLBVLD::myLBVLD() {

	if (IsOverLapping == 1) {
		block_stride = block_r;
	}
	else {
		block_stride = block_r * 2 + 1;   //It would be as same as no sliding window when block_stride equals to block_r*2+1.
	}

	Block_Size = cv::Size(2 * block_r + 1, 2 * block_r + 1);

	Block_num = cv::Size(((img_size.width - 2 * r) - Block_Size.width + block_stride) / block_stride, ((img_size.height - 2 * r) - Block_Size.height + block_stride) / block_stride);

	//Dimension of PDM matrix：D1*D2
	D1 = trainnum*classnum*Block_Size.area();// Sample_size*Sample_size - 1;       //Dimension of the W:D1*d1;

	D2 = (2 * r + 1)*(2 * r + 1) - 1;               //Dimension of the V:D2*d2;
}

void myLBVLD::Extract_PDM() {

	for (int i = 1; i<classnum + 1; i++) {
		for (int j = 1; j<trainnum + 1; j++) {

			std::string imgname = std::to_string(i) + "_" + std::to_string(j);

			cv::Mat img = cv::imread(filepath + imgname + ".jpg", 1);
			assert(!img.empty());

			cv::resize(img, img, img_size);

			if (img.channels() != 1) {
				cv::cvtColor(img, img, CV_RGB2GRAY);
			}

			train_images.push_back(img);
			img.release();
		}
	}

	std::vector<cv::Mat> PDM_vec;
	for (int m = 0; m < Block_num.height; m++) {
		for (int n = 0; n < Block_num.width; n++) {
			cv::Mat PDM = Create_PDM(m, n);

			PDM_vec.push_back(PDM);

			PDM.release();
		}
	}

	cv::FileStorage fs("PDM_vec.xml", cv::FileStorage::WRITE);
	fs << "PDM_vec" << PDM_vec;
	fs.release();
	PDM_vec.clear();

	train_images.clear();
}


cv::Mat myLBVLD::Create_PDM(int m, int n) {

	cv::Mat PDM = cv::Mat::zeros(D1, D2, CV_32FC1);

	//The first address of a pixel in a block;
	int row = m*block_stride + r;
	int col = n*block_stride + r;

	int k = 0;
	for (int i = 0; i<train_images.size(); i++) {
		//Traverse each training picture to extract the PDV matrix;

		for (int r = row; r<row + Block_Size.height; r++) {
			for (int c = col; c<col + Block_Size.width; c++) {
				//Extract the PDV matrix by traversing each pixel in a block

				cv::Mat pdv = cv::Mat::zeros(1, D2, CV_32FC1);

				for (int j = 0; j<D2; j++) {
					pdv.at<float>(0, j) = train_images[i].at<uchar>(r + table_r[j][0], c + table_r[j][1]) - train_images[i].at<uchar>(r, c);
				}

				pdv.copyTo(PDM.row(k));
				k++;
			}
		}
	}

	return PDM;
}


void myLBVLD::DFD_Train() {

	std::vector<cv::Mat> PDM_vec;
	cv::FileStorage fs;
	fs.open("PDM_vec.xml", cv::FileStorage::READ);
	fs["PDM_vec"] >> PDM_vec;
	fs.release();

	assert(PDM_vec.size() == Block_num.area());

	for (int i = 0; i<Block_num.area(); i++) {
		_2D_LDA_new(PDM_vec[i]);
		//myNet(PDM_vec[i]);
		//CreateNet(PDM_vec[i]);
		std::cout << i << std::endl;
	}

	fs.open("wv_res.xml", cv::FileStorage::WRITE);
	fs <<"W"<< W;
	fs<<"V" << V;
	fs.release();

	/*
	fs.open("wv_res_1.xml", cv::FileStorage::WRITE);
	fs << "W1" << W1;
	fs << "V1" << V1;
	fs.release();

	fs.open("wv_res_2.xml", cv::FileStorage::WRITE);
	fs << "W2" << W2;
	fs << "V2" << V2;
	fs.release();

	fs.open("wv_res_3.xml", cv::FileStorage::WRITE);
	fs << "W3" << W3;
	fs << "V3" << V3;
	fs.release();
	*/

}

void myLBVLD::cluster() {

	cv::FileStorage fs("wv_res_1.xml", cv::FileStorage::READ);
	fs["W1"] >> W1;
	fs["V1"] >> V1;
	fs.release();
	fs.open("wv_res_2.xml", cv::FileStorage::READ);
	fs["W2"] >> W2;
	fs["V2"] >> V2;
	fs.release();
	fs.open("wv_res_3.xml", cv::FileStorage::READ);
	fs["W3"] >> W3;
	fs["V3"] >> V3;
	fs.release();

	std::vector<cv::Mat> PDM_vec;
	//cv::FileStorage fs;
	fs.open("PDM_vec.xml", cv::FileStorage::READ);
	fs["PDM_vec"] >> PDM_vec;
	fs.release();

	assert(PDM_vec.size() == Block_num.area());

	std::vector<cv::Mat> img_1;
	std::vector<cv::Mat> img_2;
	std::vector<cv::Mat> img_3;

	for (int j = 0; j<Block_num.area(); j++) {
		std::vector<cv::Mat> PDM_img;
		PDM_img.reserve(trainnum*classnum);
		for (int i = 0; i<trainnum*classnum; i++) {
			cv::Mat PDM = cv::Mat::zeros(Block_Size.area(), D2, CV_32FC1);

			cv::Rect rect(0, i*Block_Size.area(), D2, Block_Size.area());

			PDM_vec[j](rect).copyTo(PDM);

			PDM_img.push_back(PDM);
		}

		for (int ttt = 0; ttt < PDM_img.size(); ttt ++ ) {
			cv::Mat img_temp1 = W1[j].t()*PDM_img[ttt] * V1[j];
			img_temp1 = sign_B(img_temp1);
			img_1.push_back(img_temp1);

			cv::Mat img_temp2 = W2[j].t()*img_temp1*V2[j];
			img_temp2 = sign_B(img_temp2);
			img_2.push_back(img_temp2);

			cv::Mat img_temp3 = W3[j].t()*img_temp2*V3[j];
			img_temp3 = sign_B(img_temp3);
			img_3.push_back(img_temp3);

		}
	}
	PDM_vec.clear();
	int rows = img_1[0].rows;
	int cols = img_1[0].cols;

	cv::Mat sum_pdv1 = cv::Mat::zeros(img_1.size()*rows, cols, CV_32FC1);
	for (int i = 0; i < img_1.size(); i++) {
		cv::Rect rect(0, i*rows, cols, rows);
		img_1[i].copyTo(sum_pdv1(rect));
	}
	cv::Mat label1;
	img_1.clear();
	cv::kmeans(sum_pdv1, 200, label1, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 0.5), 3, cv::KMEANS_PP_CENTERS, codebook1);
	fs.open("codebook1.xml", cv::FileStorage::WRITE);
	fs << "codebook1" << codebook1;

	rows = img_2[0].rows;
	cols = img_2[0].cols;
	cv::Mat sum_pdv2 = cv::Mat::zeros(img_2.size()*rows, cols, CV_32FC1);
	for (int i = 0; i < img_2.size(); i++) {
		cv::Rect rect(0, i*rows, cols, rows);
		img_2[i].copyTo(sum_pdv2(rect));
	}
	cv::Mat label2;
	img_2.clear();
	cv::kmeans(sum_pdv2, 200, label2, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 0.5), 3, cv::KMEANS_PP_CENTERS, codebook2);
	fs.open("codebook2.xml", cv::FileStorage::WRITE);
	fs << "codebook2" << codebook2;
	std::cout << "Clustering done" << std::endl;

	rows = img_3[0].rows;
	cols = img_3[0].cols;
	cv::Mat sum_pdv3 = cv::Mat::zeros(img_3.size()*rows, cols, CV_32FC1);
	for (int i = 0; i < img_3.size(); i++) {
		cv::Rect rect(0, i*rows, cols, rows);
		img_3[i].copyTo(sum_pdv3(rect));
	}
	cv::Mat label3;
	img_3.clear();
	cv::kmeans(sum_pdv3, 200, label3, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 0.5), 3, cv::KMEANS_PP_CENTERS, codebook3);
	fs.open("codebook3.xml", cv::FileStorage::WRITE);
	fs << "codebook3" << codebook3;
	std::cout << "Clustering done" << std::endl;
}

void myLBVLD::Compute_DFDVec_KMeans() {
	//Generate DFD feature vectors based on the learned w, v features,
	//and store the training pictures and the vectors generated by the test pictures separately

	cv::FileStorage fs("wv_res_1.xml", cv::FileStorage::READ);
	fs["W1"] >> W1;
	fs["V1"] >> V1;
	fs.release();
	fs.open("wv_res_2.xml", cv::FileStorage::READ);
	fs["W2"] >> W2;
	fs["V2"] >> V2;
	fs.release();
	fs.open("wv_res_3.xml", cv::FileStorage::READ);
	fs["W3"] >> W3;
	fs["V3"] >> V3;
	fs.release();


	fs.open("codebook1.xml", cv::FileStorage::READ);
	fs["codebook1"] >> codebook1;
	fs.release();

	fs.open("codebook2.xml", cv::FileStorage::READ);
	fs["codebook2"] >> codebook2;
	fs.release();

	fs.open("codebook3.xml", cv::FileStorage::READ);
	fs["codebook3"] >> codebook3;
	fs.release();

	//Generate the feature matrix of the training picture;
	for (int i = 1; i < classnum + 1; i++) {
		std::thread t1(&myLBVLD::duoxianchenglala, this, i);
		i++;
		std::thread t2(&myLBVLD::duoxianchenglala, this, i);
		i++;
		std::thread t3(&myLBVLD::duoxianchenglala, this, i);
		i++;
		std::thread t4(&myLBVLD::duoxianchenglala, this, i);
		i++;
		std::thread t5(&myLBVLD::duoxianchenglala, this, i);
		i++;
		std::thread t6(&myLBVLD::duoxianchenglala, this, i);
		t1.join();
		t2.join();
		t3.join();
		t4.join();
		t5.join();
		t6.join();
	}
}

void myLBVLD::duoxianchenglala(int i) {
	int dim1 = codebook1.rows;
	int dim2 = codebook2.rows;
	int dim3 = codebook3.rows;

	cv::Mat featureVec = cv::Mat::zeros(samplenum, Block_num.area()*(dim1+dim2+dim3), CV_32FC1);
	for (int j = 1; j<samplenum + 1; j++) {

		std::string imgname = std::to_string(i) + "_" + std::to_string(j);

		cv::Mat img = cv::imread(filepath + imgname + ".jpg", 1);
		assert(!img.empty());
		//std::cout << filepath + imgname + ".jpg" << std::endl;

		if (img.channels() != 1) {
			cv::cvtColor(img, img, CV_RGB2GRAY);
		}

		cv::resize(img, img, img_size);

		cv::Mat mat = extractPDM_KMeans(img);
		mat.copyTo(featureVec.row(j - 1));

		//std::cout<<i<<"  "<<j<<std::endl;

	}
	cv::FileStorage fs;
	fs.open("Res\\" + std::to_string(i) + ".xml", cv::FileStorage::WRITE);
	fs << "featureVec" << featureVec;
	fs.release();

	std::cout << "*********" << i << "***********" << std::endl;
}

cv::Mat myLBVLD::extractPDM_KMeans(cv::Mat img) {
	std::vector<cv::Mat> PDMs;

	for (int m = 0; m < Block_num.height; m++) {
		for (int n = 0; n < Block_num.width; n++) {

			cv::Mat PDM = cv::Mat::zeros(Block_Size.area(), D2, CV_32FC1);
			//The first address of a pixel in a block;
			int row = m * block_stride + r;
			int col = n * block_stride + r;

			int k = 0;
			for (int r = row; r<row + Block_Size.height; r++) {
				for (int c = col; c<col + Block_Size.width; c++) {
					//Extract the PDV matrix by traversing each pixel in a block

					cv::Mat pdv = cv::Mat::zeros(1, D2, CV_32FC1);
					for (int j = 0; j<D2; j++) {
						pdv.at<float>(0, j) = img.at<uchar>(r + table_r[j][0], c + table_r[j][1]) - img.at<uchar>(r, c);
					}

					pdv.copyTo(PDM.row(k));
					k++;
					pdv.release();
				}
			}

			PDMs.push_back(PDM);

		}
	}

	assert(PDMs.size() == W1.size() && W1[0].rows == PDMs[0].rows&&PDMs[0].cols == V1[0].rows);

	//int dim = pow(2, V[0].cols);
	//cv::Mat featureVec = cv::Mat::zeros(1, PDMs.size()*dim, CV_32FC1);
	int dim1 = codebook1.rows;
	cv::Mat featureMat = cv::Mat::zeros(PDMs.size(), dim1*3, CV_32SC1);
	for (int i = 0; i<PDMs.size(); i++) {

		cv::Mat temp1 = W1[i].t()*PDMs[i] * V1[i];
		cv::Mat temp2 = W2[i].t()*temp1*V2[i];
		cv::Mat temp3 = W3[i].t()*temp2*V3[i];
		temp1 = sign_B(temp1);
		temp2 = sign_B(temp2);
		temp3 = sign_B(temp3);

		std::vector<int> feature1(dim1,0);
		for (int zz = 0; zz < temp1.rows; zz++) {
			int tag=compute_Dis(temp1.row(zz),codebook1);
			feature1.at(tag)++;
		}

		std::vector<int> feature2(dim1, 0);
		for (int zz = 0; zz < temp2.rows; zz++) {
			int tag = compute_Dis(temp2.row(zz), codebook2);
			feature2.at(tag)++;
		}

		std::vector<int> feature3(dim1, 0);
		for (int zz = 0; zz < temp3.rows; zz++) {
			int tag = compute_Dis(temp3.row(zz), codebook3);
			feature3.at(tag)++;
		}

		std::copy(feature1.begin(), feature1.end(), featureMat.row(i).begin<int>());
		std::copy(feature2.begin(), feature2.end(), featureMat.row(i).begin<int>()+dim1);
		std::copy(feature3.begin(), feature3.end(), featureMat.row(i).begin<int>()+dim1*2);
	}
	//std::vector<int> featureVec(featureMat.rows*featureMat.cols, 0);
	cv::Mat featureVec = cv::Mat::zeros(1, featureMat.rows*featureMat.cols, CV_32SC1);
	std::copy(featureMat.begin<int>(), featureMat.end<int>(), featureVec.begin<int>());

	cv::Mat Vec = NormalizeVec(featureVec);

	return Vec;
}

int myLBVLD::compute_Dis(cv::Mat A, cv::Mat codebook) {

	assert(A.rows == 1 && A.cols == codebook.cols);
	float mindis = 999999999.0;
	int tag = codebook.rows + 1;

	for (int i = 0; i < codebook.rows; i++) {
		cv::Mat temp = codebook.row(i) - A;
		temp = temp * temp.t();
		float dis_temp = temp.at<float>(0, 0);
		if (dis_temp < mindis) {
			mindis = dis_temp;
			tag = i;
		}
	}
	return tag;
}

std::vector<cv::Mat> myLBVLD::learn_w_and_v(std::vector<cv::Mat>PDM_img, cv::Mat  &w, cv::Mat &v) {


	//Calculate the mean of each class and the mean of each picture;
	cv::Mat sum_mean;   //The mean of all training sets;
	std::vector<cv::Mat> class_mean_vec;   //Store the matrix mean of each category;

	sum_mean = cv::Mat::zeros(PDM_img[0].rows, PDM_img[0].cols, CV_32FC1);
	for (int i = 0; i<classnum; i++) {
		cv::Mat temp = cv::Mat::zeros(PDM_img[0].rows, PDM_img[0].cols, CV_32FC1);
		for (int j = 0; j<trainnum; j++) {
			int tag = i * trainnum + j;
			temp += PDM_img[tag];
			sum_mean += PDM_img[tag];
		}
		temp = temp / trainnum;
		class_mean_vec.push_back(temp);
	}
	sum_mean = sum_mean / (classnum*trainnum);

	cv::Mat Sb1 = cv::Mat::zeros(sum_mean.rows, sum_mean.rows, CV_32FC1);
	cv::Mat Sw1 = cv::Mat::zeros(sum_mean.rows, sum_mean.rows, CV_32FC1);

	for (int i = 0; i<classnum; i++) {
		for (int j = 0; j<trainnum; j++) {
			int tag = i * trainnum + j;
			Sw1 += (PDM_img[tag] - class_mean_vec[i])*v*v.t()*(PDM_img[tag] - class_mean_vec[i]).t();
		}

		Sb1 += (class_mean_vec[i] - sum_mean)*v*v.t()*(class_mean_vec[i] - sum_mean).t();
	}
	Sb1 = Sb1 * trainnum;

	//std::cout << Sw1 << std::endl;

	cv::Mat eigenValue_1, eigenVector_1;
	cv::eigen(Sw1.inv()*Sb1, eigenValue_1, eigenVector_1);
	//cv::PCA pca(Sw1.inv()*Sb1,cv::Mat(),CV_PCA_DATA_AS_ROW,w_cols);
	assert(eigenVector_1.cols == w.rows&&eigenVector_1.rows >= w.cols);
	cv::Rect rect1(0, 0, eigenVector_1.cols, w.cols);
	//w.release();
	w = eigenVector_1(rect1).t();
	Sb1.release();
	Sw1.release();
	eigenValue_1.release();
	eigenVector_1.release();


	cv::Mat Sw2 = cv::Mat::zeros(sum_mean.cols, sum_mean.cols, CV_32FC1);
	cv::Mat Sb2 = cv::Mat::zeros(sum_mean.cols, sum_mean.cols, CV_32FC1);

	for (int i = 0; i<classnum; i++) {
		for (int j = 0; j<trainnum; j++) {
			int tag = i * trainnum + j;
			Sw2 += (PDM_img[tag] - class_mean_vec[i]).t()*w*w.t()*(PDM_img[tag] - class_mean_vec[i]);
		}
		Sb2 += (class_mean_vec[i] - sum_mean).t()*w*w.t()*(class_mean_vec[i] - sum_mean);
	}
	Sb2 = Sb2 * trainnum;

	cv::Mat eigenValue_2, eigenVector_2;
	cv::eigen(Sw2.inv()*Sb2, eigenValue_2, eigenVector_2);
	//cv::PCA pca2(Sw2.inv()*Sb2,cv::Mat(),CV_PCA_DATA_AS_ROW,v_cols);
	assert(eigenVector_2.cols == v.rows&&eigenVector_2.rows >= v.cols);
	cv::Rect rect2(0, 0, eigenVector_2.cols, v.cols);
	//v.release();
	v = eigenVector_2(rect2).t();
	Sb2.release();
	Sw2.release();
	eigenValue_2.release();
	eigenVector_2.release();

	std::vector<cv::Mat> PDMs_after;
	for (int i = 0; i < PDM_img.size(); i++) {
		cv::Mat temp_img = w.t()*PDM_img[i] * v;
		PDMs_after.push_back(temp_img);
	}

	return PDMs_after;
}

void myLBVLD::CreateNet(cv::Mat PDMs) {

	//cv::Mat w = cv::Mat::eye(sum_mean.rows, w_cols, CV_32FC1);
	//cv::Mat v = cv::Mat::eye(sum_mean.cols, v_cols, CV_32FC1);

	std::cout << "**********start the net!!!!************" << std::endl;
	//Divide PDMs into pictures;
	std::vector<cv::Mat> PDM_img;
	PDM_img.reserve(trainnum*classnum);

	for (int i = 0; i<trainnum*classnum; i++) {
		cv::Mat PDM = cv::Mat::zeros(Block_Size.area(), D2, CV_32FC1);

		cv::Rect rect(0, i*Block_Size.area(), D2, Block_Size.area());

		PDMs(rect).copyTo(PDM);

		PDM_img.push_back(PDM);
	}


	float Jmin =0, Jmin_old=999999999.1;
	cv::Mat w1, v1, w2, v2, w3, v3;

	w1 = cv::Mat::eye(PDM_img[0].rows, 44, CV_32FC1);
	v1 = cv::Mat::eye(PDM_img[0].cols, 32, CV_32FC1);
	std::vector<cv::Mat>PDM_img1 = learn_w_and_v(PDM_img, w1, v1);
	Jmin += Createloss(PDM_img, w1, v1);
	
	w2 = cv::Mat::eye(PDM_img1[0].rows, 40, CV_32FC1);
	v2 = cv::Mat::eye(PDM_img1[0].cols, 16, CV_32FC1);
	std::vector<cv::Mat>PDM_img2 = learn_w_and_v(PDM_img1, w2, v2);
	Jmin += Createloss(PDM_img1, w2, v2);

	w3 = cv::Mat::eye(PDM_img2[0].rows, 36, CV_32FC1);
	v3 = cv::Mat::eye(PDM_img2[0].cols, 8, CV_32FC1);
	learn_w_and_v(PDM_img2, w3, v3);

	Jmin += Createloss(PDM_img2, w3, v3)+loss_2(PDM_img2,w3,v3);

	while (Jmin < Jmin_old) {
		Jmin_old = Jmin;
		Jmin = 0;


		PDM_img1 = learn_w_and_v(PDM_img, w1, v1);
		Jmin += Createloss(PDM_img, w1, v1);

		PDM_img2 = learn_w_and_v(PDM_img1, w2, v2);
		Jmin += Createloss(PDM_img1, w2, v2);

		learn_w_and_v(PDM_img2, w3, v3);
		Jmin += Createloss(PDM_img2, w3, v3)+ loss_2(PDM_img2, w3, v3);
	}


	W1.push_back(w1);
	V1.push_back(v1);


	W2.push_back(w2);
	V2.push_back(v2);

	W3.push_back(w3);
	V3.push_back(v3);


}

float myLBVLD::loss_2(std::vector<cv::Mat>PDM_img, cv::Mat  w, cv::Mat v) {

	cv::Mat D_w, D_b;
	D_w = cv::Mat::zeros(w.cols, w.cols, CV_32FC1);
	D_b = cv::Mat::zeros(w.cols, w.cols, CV_32FC1);

	for (int i = 0; i < PDM_img.size(); i++) {
		cv::Mat temp_mat = w.t()*PDM_img[i] * v;
		D_w += (sign_B(temp_mat) - Compute_Mean_of_Matrix(sign_B(temp_mat)))*(sign_B(temp_mat) - Compute_Mean_of_Matrix(sign_B(temp_mat))).t();
		D_b += (sign_B(temp_mat) - temp_mat)*(sign_B(temp_mat) - temp_mat).t();
	}
	float loss = cv::trace(D_b).val[0] - cv::trace(D_w).val[0];
	return loss;
}

float myLBVLD::Createloss(std::vector<cv::Mat>PDM_img, cv::Mat  w, cv::Mat v) {

	cv::Mat sum_mean;   //The mean of all training sets;
	std::vector<cv::Mat> class_mean_vec;   //Matrix mean for each class;

	sum_mean = cv::Mat::zeros(PDM_img[0].rows, PDM_img[0].cols, CV_32FC1);
	for (int i = 0; i<classnum; i++) {
		cv::Mat temp = cv::Mat::zeros(PDM_img[0].rows, PDM_img[0].cols, CV_32FC1);
		for (int j = 0; j<trainnum; j++) {
			int tag = i * trainnum + j;
			temp += PDM_img[tag];
			sum_mean += PDM_img[tag];
		}
		temp = temp / trainnum;
		class_mean_vec.push_back(temp);
	}
	sum_mean = sum_mean / (classnum*trainnum);

	cv::Mat D_w, D_b;
	D_w = cv::Mat::zeros(w.cols, w.cols, CV_32FC1);
	D_b = cv::Mat::zeros(w.cols, w.cols, CV_32FC1);

	//float tr=0;
	for (int i = 0; i<classnum; i++) {
		for (int j = 0; j<trainnum; j++) {
			int tag = i * trainnum + j;
			D_w += w.t()*(PDM_img[tag] - class_mean_vec[i])*v*v.t()*(PDM_img[tag] - class_mean_vec[i]).t()*w;

		}
		D_b += w.t()*(class_mean_vec[i] - sum_mean)*v*v.t()*(class_mean_vec[i] - sum_mean).t()*w;
	}
	D_b = D_b * trainnum;
	//Jmin=sita_w*(cv::trace(D_w).val[0])-sita_b*(cv::trace(D_b).val[0])+sita_B*tr;
	float loss = sita_w * (cv::trace(D_w).val[0]) - sita_b * (cv::trace(D_b).val[0]);
	return loss;
}


//2D_LDA modified version 1, adding D_w and D_b;
std::vector<cv::Mat> myLBVLD::_2D_LDA(std::vector<cv::Mat> PDM_img, int w_cols, int v_cols, int tag) {

	//Calculate the mean of each category and the mean of each picture;
	cv::Mat sum_mean;   ////The mean of all training sets;
	std::vector<cv::Mat> class_mean_vec;   //Matrix mean for each class;

	sum_mean = cv::Mat::zeros(PDM_img[0].rows, PDM_img[0].cols, CV_32FC1);
	for (int i = 0; i<classnum; i++) {
		cv::Mat temp = cv::Mat::zeros(PDM_img[0].rows, PDM_img[0].cols, CV_32FC1);
		for (int j = 0; j<trainnum; j++) {
			int tag = i * trainnum + j;
			temp += PDM_img[tag];
			sum_mean += PDM_img[tag];
		}
		temp = temp / trainnum;
		class_mean_vec.push_back(temp);
	}
	sum_mean = sum_mean / (classnum*trainnum);

	//Perform iterative training to generate w, v;
	//int w_cols = Block_Size.area() * 3 / 4;      //***************
	//int v_cols = 8;                         //***************

	//assert(PDM_img[0].rows == sum_mean.rows&&PDM_img[0].cols == sum_mean.cols);
	//assert(PDM_img[0].rows == class_mean_vec[0].rows&&PDM_img[0].cols == class_mean_vec[0].cols);

	cv::Mat w = cv::Mat::eye(sum_mean.rows, w_cols, CV_32FC1);
	cv::Mat v = cv::Mat::eye(sum_mean.cols, v_cols, CV_32FC1);

	float Jmin = 9999999, Jmin_old;
	do {
		Jmin_old = Jmin;

		//Fixed v, update iteratively w
		cv::Mat Sb1 = cv::Mat::zeros(sum_mean.rows, sum_mean.rows, CV_32FC1);
		cv::Mat Sw1 = cv::Mat::zeros(sum_mean.rows, sum_mean.rows, CV_32FC1);

		for (int i = 0; i<classnum; i++) {
			for (int j = 0; j<trainnum; j++) {
				int tag = i * trainnum + j;
				Sw1 += (PDM_img[tag] - class_mean_vec[i])*v*v.t()*(PDM_img[tag] - class_mean_vec[i]).t();
			}

			Sb1 += (class_mean_vec[i] - sum_mean)*v*v.t()*(class_mean_vec[i] - sum_mean).t();
		}
		Sb1 = Sb1 * trainnum;

		//std::cout << Sw1 << std::endl;

		cv::Mat eigenValue_1, eigenVector_1;
		cv::eigen(Sw1.inv()*Sb1, eigenValue_1, eigenVector_1);
		//cv::PCA pca(Sw1.inv()*Sb1,cv::Mat(),CV_PCA_DATA_AS_ROW,w_cols);
		assert(eigenVector_1.cols == w.rows&&eigenVector_1.rows >= w.cols);
		cv::Rect rect1(0, 0, eigenVector_1.cols, w_cols);
		w.release();
		w = eigenVector_1(rect1).t();

		Sb1.release();
		Sw1.release();
		eigenValue_1.release();
		eigenVector_1.release();

		//Fixed w, update v
		cv::Mat Sw2 = cv::Mat::zeros(sum_mean.cols, sum_mean.cols, CV_32FC1);
		cv::Mat Sb2 = cv::Mat::zeros(sum_mean.cols, sum_mean.cols, CV_32FC1);

		for (int i = 0; i<classnum; i++) {
			for (int j = 0; j<trainnum; j++) {
				int tag = i * trainnum + j;
				Sw2 += (PDM_img[tag] - class_mean_vec[i]).t()*w*w.t()*(PDM_img[tag] - class_mean_vec[i]);
			}
			Sb2 += (class_mean_vec[i] - sum_mean).t()*w*w.t()*(class_mean_vec[i] - sum_mean);
		}
		Sb2 = Sb2 * trainnum;

		cv::Mat eigenValue_2, eigenVector_2;
		cv::eigen(Sw2.inv()*Sb2, eigenValue_2, eigenVector_2);
		//cv::PCA pca2(Sw2.inv()*Sb2,cv::Mat(),CV_PCA_DATA_AS_ROW,v_cols);
		assert(eigenVector_2.cols == v.rows&&eigenVector_2.rows >= v.cols);
		cv::Rect rect2(0, 0, eigenVector_2.cols, v_cols);
		v.release();
		v = eigenVector_2(rect2).t();

		Sb2.release();
		Sw2.release();
		eigenValue_2.release();
		eigenVector_2.release();

		cv::Mat D_w, D_b;
		D_w = cv::Mat::zeros(w_cols, w_cols, CV_32FC1);
		D_b = cv::Mat::zeros(w_cols, w_cols, CV_32FC1);

		//float tr=0;
		for (int i = 0; i<classnum; i++) {
			for (int j = 0; j<trainnum; j++) {
				int tag = i * trainnum + j;
				D_w += w.t()*(PDM_img[tag] - class_mean_vec[i])*v*v.t()*(PDM_img[tag] - class_mean_vec[i]).t()*w;

			}
			D_b += w.t()*(class_mean_vec[i] - sum_mean)*v*v.t()*(class_mean_vec[i] - sum_mean).t()*w;
		}
		D_b = D_b * trainnum;
		//Jmin=sita_w*(cv::trace(D_w).val[0])-sita_b*(cv::trace(D_b).val[0])+sita_B*tr;
		Jmin = sita_w * (cv::trace(D_w).val[0]) - sita_b * (cv::trace(D_b).val[0]);


	} while (Jmin_old>Jmin);

	std::vector<cv::Mat> PDMs_after;
	for (int i = 0; i < PDM_img.size(); i++) {
		cv::Mat temp_img = w.t()*PDM_img[i] * v;
		PDMs_after.push_back(temp_img);
	}

	W.push_back(w);

	V.push_back(v);

	if (tag == 1) {
		W1.push_back(w);
		V1.push_back(v);
	}
	else if (tag == 2) {
		W2.push_back(w);
		V2.push_back(v);
	}
	else {
		W3.push_back(w);
		V3.push_back(v);
	}

	return PDMs_after;
}

//2D_LDA modified version 1, adding D_w and D_b;
void myLBVLD::_2D_LDA(cv::Mat PDMs) {

	//Divide PDMs into pictures;
	std::vector<cv::Mat> PDM_img;
	PDM_img.reserve(trainnum*classnum);

	for (int i = 0; i<trainnum*classnum; i++) {
		cv::Mat PDM = cv::Mat::zeros(Block_Size.area(), D2, CV_32FC1);

		cv::Rect rect(0, i*Block_Size.area(), D2, Block_Size.area());

		PDMs(rect).copyTo(PDM);

		PDM_img.push_back(PDM);
	}

	//Calculate the mean of each category and the mean of each picture;
	cv::Mat sum_mean;   //The mean of all training sets;
	std::vector<cv::Mat> class_mean_vec;   //Matrix mean for each category;
	sum_mean = cv::Mat::zeros(Block_Size.area(), D2, CV_32FC1);
	for (int i = 0; i<classnum; i++) {
		cv::Mat temp = cv::Mat::zeros(Block_Size.area(), D2, CV_32FC1);
		for (int j = 0; j<trainnum; j++) {
			int tag = i*trainnum + j;
			temp += PDM_img[tag];
			sum_mean += PDM_img[tag];
		}
		temp = temp / trainnum;
		class_mean_vec.push_back(temp);
	}
	sum_mean = sum_mean / (classnum*trainnum);

	//Perform iterative training to generate w, v;
	int w_cols = Block_Size.area()* 3 / 4;      //***************
	int v_cols = 8;                         //***************

	assert(PDM_img[0].rows == sum_mean.rows&&PDM_img[0].cols == sum_mean.cols);
	assert(PDM_img[0].rows == class_mean_vec[0].rows&&PDM_img[0].cols == class_mean_vec[0].cols);

	cv::Mat w = cv::Mat::eye(sum_mean.rows, w_cols, CV_32FC1);
	cv::Mat v = cv::Mat::eye(sum_mean.cols, v_cols, CV_32FC1);

	float Jmin = 9999999, Jmin_old;
	do {
		Jmin_old = Jmin;

		//Fixed v, update iteratively w
		cv::Mat Sb1 = cv::Mat::zeros(sum_mean.rows, sum_mean.rows, CV_32FC1);
		cv::Mat Sw1 = cv::Mat::zeros(sum_mean.rows, sum_mean.rows, CV_32FC1);

		for (int i = 0; i<classnum; i++) {
			for (int j = 0; j<trainnum; j++) {
				int tag = i*trainnum + j;
				Sw1 += (PDM_img[tag] - class_mean_vec[i])*v*v.t()*(PDM_img[tag] - class_mean_vec[i]).t();
			}

			Sb1 += (class_mean_vec[i] - sum_mean)*v*v.t()*(class_mean_vec[i] - sum_mean).t();
		}
		Sb1 = Sb1*trainnum;

		//std::cout << Sw1 << std::endl;

		cv::Mat eigenValue_1, eigenVector_1;
		cv::eigen(Sw1.inv()*Sb1, eigenValue_1, eigenVector_1);
		//cv::PCA pca(Sw1.inv()*Sb1,cv::Mat(),CV_PCA_DATA_AS_ROW,w_cols);
		assert(eigenVector_1.cols == w.rows&&eigenVector_1.rows >= w.cols);
		cv::Rect rect1(0, 0, eigenVector_1.cols, w_cols);
		w.release();
		w = eigenVector_1(rect1).t();

		Sb1.release();
		Sw1.release();
		eigenValue_1.release();
		eigenVector_1.release();

		//Fixed w, update iteratively v
		cv::Mat Sw2 = cv::Mat::zeros(sum_mean.cols, sum_mean.cols, CV_32FC1);
		cv::Mat Sb2 = cv::Mat::zeros(sum_mean.cols, sum_mean.cols, CV_32FC1);

		for (int i = 0; i<classnum; i++) {
			for (int j = 0; j<trainnum; j++) {
				int tag = i*trainnum + j;
				Sw2 += (PDM_img[tag] - class_mean_vec[i]).t()*w*w.t()*(PDM_img[tag] - class_mean_vec[i]);
			}
			Sb2 += (class_mean_vec[i] - sum_mean).t()*w*w.t()*(class_mean_vec[i] - sum_mean);
		}
		Sb2 = Sb2*trainnum;

		cv::Mat eigenValue_2, eigenVector_2;
		cv::eigen(Sw2.inv()*Sb2, eigenValue_2, eigenVector_2);
		//cv::PCA pca2(Sw2.inv()*Sb2,cv::Mat(),CV_PCA_DATA_AS_ROW,v_cols);
		assert(eigenVector_2.cols == v.rows&&eigenVector_2.rows >= v.cols);
		cv::Rect rect2(0, 0, eigenVector_2.cols, v_cols);
		v.release();
		v = eigenVector_2(rect2).t();

		Sb2.release();
		Sw2.release();
		eigenValue_2.release();
		eigenVector_2.release();

		cv::Mat D_w, D_b;
		D_w = cv::Mat::zeros(w_cols, w_cols, CV_32FC1);
		D_b = cv::Mat::zeros(w_cols, w_cols, CV_32FC1);

		//float tr=0;
		for (int i = 0; i<classnum; i++) {
			for (int j = 0; j<trainnum; j++) {
				int tag = i*trainnum + j;
				D_w += w.t()*(PDM_img[tag] - class_mean_vec[i])*v*v.t()*(PDM_img[tag] - class_mean_vec[i]).t()*w;

			}
			D_b += w.t()*(class_mean_vec[i] - sum_mean)*v*v.t()*(class_mean_vec[i] - sum_mean).t()*w;
		}
		D_b = D_b*trainnum;
		//Jmin=sita_w*(cv::trace(D_w).val[0])-sita_b*(cv::trace(D_b).val[0])+sita_B*tr;
		Jmin = sita_w*(cv::trace(D_w).val[0]) - sita_b*(cv::trace(D_b).val[0]);


	} while (Jmin_old>Jmin);

	W.push_back(w);
	V.push_back(v);

}

void myLBVLD::_2D_LDA_new(cv::Mat PDMs) {

	//Divide PDMs into pictures;
	std::vector<cv::Mat> PDM_img;
	PDM_img.reserve(trainnum*classnum);

	for (int i = 0; i<trainnum*classnum; i++) {
		cv::Mat PDM = cv::Mat::zeros(Block_Size.area(), D2, CV_32FC1);

		cv::Rect rect(0, i*Block_Size.area(), D2, Block_Size.area());

		PDMs(rect).copyTo(PDM);

		PDM_img.push_back(PDM);
	}

	//Calculate the mean of each category and the mean of each picture;
	cv::Mat sum_mean;   //The mean of all training sets;
	std::vector<cv::Mat> class_mean_vec;   //Matrix mean for each category;

	sum_mean = cv::Mat::zeros(Block_Size.area(), D2, CV_32FC1);
	for (int i = 0; i<classnum; i++) {
		cv::Mat temp = cv::Mat::zeros(Block_Size.area(), D2, CV_32FC1);
		for (int j = 0; j<trainnum; j++) {
			int tag = i*trainnum + j;
			temp += PDM_img[tag];
			sum_mean += PDM_img[tag];
		}
		temp = temp / trainnum;
		class_mean_vec.push_back(temp);
	}
	sum_mean = sum_mean / (classnum*trainnum);

	//Perform iterative training to generate w, v;
	int w_cols = Block_Size.area() * 3 / 4;      //***************
	int v_cols = 8;                         //***************

	assert(PDM_img[0].rows == sum_mean.rows&&PDM_img[0].cols == sum_mean.cols);
	assert(PDM_img[0].rows == class_mean_vec[0].rows&&PDM_img[0].cols == class_mean_vec[0].cols);

	cv::Mat w = cv::Mat::eye(sum_mean.rows, w_cols, CV_32FC1);
	cv::Mat v = cv::Mat::eye(sum_mean.cols, v_cols, CV_32FC1);

	float Jmin = 9999999, Jmin_old;
	do {
		Jmin_old = Jmin;

		//Fixed v,update w
		cv::Mat Sb1 = cv::Mat::zeros(sum_mean.rows, sum_mean.rows, CV_32FC1);
		cv::Mat Sw1 = cv::Mat::zeros(sum_mean.rows, sum_mean.rows, CV_32FC1);
		cv::Mat Sb1_temp = cv::Mat::zeros(sum_mean.rows, sum_mean.rows, CV_32FC1);

		for (int i = 0; i<classnum; i++) {
			for (int j = 0; j<trainnum; j++) {
				int tag = i*trainnum + j;
				Sw1 += (PDM_img[tag] - class_mean_vec[i])*v*v.t()*(PDM_img[tag] - class_mean_vec[i]).t()+(sign_B(PDM_img[tag]*v)-(PDM_img[tag]*v))*(sign_B(PDM_img[tag] * v) - (PDM_img[tag] * v)).t();
				Sb1_temp += ((sign_B(PDM_img[tag] * v)) - Compute_Mean_of_Matrix(sign_B(PDM_img[tag] * v)))*((sign_B(PDM_img[tag] * v)) - Compute_Mean_of_Matrix(sign_B(PDM_img[tag] * v))).t();
			}

			Sb1 += (class_mean_vec[i] - sum_mean)*v*v.t()*(class_mean_vec[i] - sum_mean).t();
		}
		Sb1 = Sb1*trainnum+Sb1_temp;

		cv::Mat eigenValue_1, eigenVector_1;
		cv::eigen(Sw1.inv()*Sb1, eigenValue_1, eigenVector_1);
		//cv::PCA pca(Sw1.inv()*Sb1,cv::Mat(),CV_PCA_DATA_AS_ROW,w_cols);
		assert(eigenVector_1.cols == w.rows&&eigenVector_1.rows >= w.cols);
		cv::Rect rect1(0, 0, eigenVector_1.cols, w_cols);
		w.release();
		w = eigenVector_1(rect1).t();

		Sb1.release();
		Sw1.release();
		Sb1_temp.release();
		eigenValue_1.release();
		eigenVector_1.release();

		//Fixed w, update v
		cv::Mat Sw2 = cv::Mat::zeros(sum_mean.cols, sum_mean.cols, CV_32FC1);
		cv::Mat Sb2 = cv::Mat::zeros(sum_mean.cols, sum_mean.cols, CV_32FC1);
		cv::Mat Sb2_temp = cv::Mat::zeros(sum_mean.cols, sum_mean.cols, CV_32FC1);

		for (int i = 0; i<classnum; i++) {
			for (int j = 0; j<trainnum; j++) {
				int tag = i*trainnum + j;
				Sw2 += (PDM_img[tag] - class_mean_vec[i]).t()*w*w.t()*(PDM_img[tag] - class_mean_vec[i])+(sign_B(PDM_img[tag].t()*w)-(PDM_img[tag].t()*w))*(sign_B(PDM_img[tag].t()*w) - (PDM_img[tag].t()*w)).t();
				Sb2_temp += ((sign_B(PDM_img[tag].t()*w)) - Compute_Mean_of_Matrix(sign_B(PDM_img[tag].t()*w)))*((sign_B(PDM_img[tag].t()*w)) - Compute_Mean_of_Matrix(sign_B(PDM_img[tag].t()*w))).t();
			}
			Sb2 += (class_mean_vec[i] - sum_mean).t()*w*w.t()*(class_mean_vec[i] - sum_mean);
		}
		Sb2 = Sb2*trainnum+Sb2_temp;

		cv::Mat eigenValue_2, eigenVector_2;
		cv::eigen(Sw2.inv()*Sb2, eigenValue_2, eigenVector_2);
		//cv::PCA pca2(Sw2.inv()*Sb2,cv::Mat(),CV_PCA_DATA_AS_ROW,v_cols);
		assert(eigenVector_2.cols == v.rows&&eigenVector_2.rows >= v.cols);
		cv::Rect rect2(0, 0, eigenVector_2.cols, v_cols);
		v.release();
		v = eigenVector_2(rect2).t();

		Sb2.release();
		Sw2.release();
		eigenValue_2.release();
		eigenVector_2.release();

		cv::Mat D_w, D_b,D_s,D_u;
		D_w = cv::Mat::zeros(w_cols, w_cols, CV_32FC1);
		D_b = cv::Mat::zeros(w_cols, w_cols, CV_32FC1);
		D_s = cv::Mat::zeros(w_cols, w_cols, CV_32FC1);
		D_u = cv::Mat::zeros(w_cols, w_cols, CV_32FC1);

		//float tr=0;
		for (int i = 0; i<classnum; i++) {
			for (int j = 0; j<trainnum; j++) {
				int tag = i*trainnum + j;
				D_w += w.t()*(PDM_img[tag] - class_mean_vec[i])*v*v.t()*(PDM_img[tag] - class_mean_vec[i]).t()*w;
				D_s += (sign_B(w.t()*PDM_img[tag] * v) - (w.t()*PDM_img[tag] * v))*(sign_B(w.t()*PDM_img[tag] * v) - (w.t()*PDM_img[tag] * v)).t();
				D_u += (sign_B(w.t()*PDM_img[tag] * v) - Compute_Mean_of_Matrix(sign_B(w.t()*PDM_img[tag] * v)))*(sign_B(w.t()*PDM_img[tag] * v) - Compute_Mean_of_Matrix(sign_B(w.t()*PDM_img[tag] * v))).t();

			}
			D_b += w.t()*(class_mean_vec[i] - sum_mean)*v*v.t()*(class_mean_vec[i] - sum_mean).t()*w;
		}
		D_b = D_b*trainnum;
		//Jmin=sita_w*(cv::trace(D_w).val[0])-sita_b*(cv::trace(D_b).val[0])+sita_B*tr;
		Jmin = sita_w*(cv::trace(D_w).val[0]) - sita_b*(cv::trace(D_b).val[0])+ sita_s*(cv::trace(D_s).val[0])-sita_u*(cv::trace(D_u).val[0]);


	} while (Jmin_old>Jmin);

	W.push_back(w);
	V.push_back(v);

}

void myLBVLD::Compute_DFDVec_new() {
	//Generate DFD feature vectors based on the learned w, v features, 
	//and store the training pictures and the vectors generated by the test pictures separately

	cv::FileStorage fs("wv_res_1.xml", cv::FileStorage::READ);
	fs["W1"] >> W1;
	fs["V1"] >> V1;
	fs.release();
	fs.open("wv_res_2.xml", cv::FileStorage::READ);
	fs["W2"] >> W2;
	fs["V2"] >> V2;
	fs.release();
	fs.open("wv_res_3.xml", cv::FileStorage::READ);
	fs["W3"] >> W3;
	fs["V3"] >> V3;
	fs.release();

	//Generate the feature matrix of the training picture;
	for (int i = 1; i<classnum + 1; i++) {

		int dim = pow(2, V3[0].cols);
		cv::Mat featureVec = cv::Mat::zeros(samplenum, Block_num.area()*dim, CV_32FC1);
		for (int j = 1; j<samplenum + 1; j++) {

			std::string imgname = std::to_string(i) + "_" + std::to_string(j);

			cv::Mat img = cv::imread(filepath + imgname + ".jpg", 1);
			assert(!img.empty());
			//std::cout << filepath + imgname + ".jpg" << std::endl;

			if (img.channels() != 1) {
				cv::cvtColor(img, img, CV_RGB2GRAY);
			}

			cv::resize(img, img, img_size);

			cv::Mat mat = Extract_PDM_new(img);
			mat.copyTo(featureVec.row(j - 1));

			//std::cout<<i<<"  "<<j<<std::endl;

		}
		std::cout << "**********" << i << std::endl;
		fs.open("Res\\" + std::to_string(i) + ".xml", cv::FileStorage::WRITE);
		fs << "featureVec" << featureVec;
		fs.release();
	}
}

cv::Mat myLBVLD::Extract_PDM_new(cv::Mat img) {
	//Extract PDM matrix from a picture

	std::vector<cv::Mat> PDMs;

	for (int m = 0; m < Block_num.height; m++) {
		for (int n = 0; n < Block_num.width; n++) {

			cv::Mat PDM = cv::Mat::zeros(Block_Size.area(), D2, CV_32FC1);
			//The first address of a pixel in a block;
			int row = m * block_stride + r;
			int col = n * block_stride + r;

			int k = 0;
			for (int r = row; r<row + Block_Size.height; r++) {
				for (int c = col; c<col + Block_Size.width; c++) {
					//Extract the PDV matrix by traversing each pixel in a block

					cv::Mat pdv = cv::Mat::zeros(1, D2, CV_32FC1);
					for (int j = 0; j<D2; j++) {
						pdv.at<float>(0, j) = img.at<uchar>(r + table_r[j][0], c + table_r[j][1]) - img.at<uchar>(r, c);
					}

					pdv.copyTo(PDM.row(k));
					k++;
					pdv.release();
				}
			}

			PDMs.push_back(PDM);

		}
	}

	assert(PDMs.size() == W1.size() && W1[0].rows == PDMs[0].rows&&PDMs[0].cols == V1[0].rows);
	assert(PDMs.size() == W2.size() );
	assert(PDMs.size() == W3.size() );

	int dim = pow(2, V3[0].cols);
	cv::Mat featureVec = cv::Mat::zeros(1, PDMs.size()*dim, CV_32FC1);
	for (int i = 0; i<PDMs.size(); i++) {

		cv::Mat temp = W1[i].t()*PDMs[i] * V1[i];
		temp = W2[i].t()*temp * V2[i];
		temp = W3[i].t()*temp* V3[i];
		cv::Rect rect(i*dim, 0, dim, 1);
		cv::Mat mat = Mat_Map_HashBinary(temp);

		cv::Mat vec = cv::Mat::zeros(1, dim, CV_32FC1);
		for (int i = 0; i<mat.cols; i++) {
			int temp = (int)mat.at<float>(0, i);
			vec.at<float>(0, temp)++;
		}
		vec.copyTo(featureVec(rect));
	}

	cv::Mat Vec = NormalizeVec(featureVec);

	return Vec;
}

void myLBVLD::Compute_DFDVec() {
	//Generate DFD feature vectors based on the learned w, v features, 
	//and store the training pictures and the vectors generated by the test pictures separately

	cv::FileStorage fs("wv_res.xml", cv::FileStorage::READ);
	fs["W"] >> W;
	fs["V"] >> V;
	fs.release();

	//Generate the feature matrix of the training picture;
	for (int i = 1; i<classnum + 1; i++) {

		int dim = pow(2, V[0].cols);
		cv::Mat featureVec = cv::Mat::zeros(samplenum, Block_num.area()*dim, CV_32FC1);
		for (int j = 1; j<samplenum + 1; j++) {

			std::string imgname = std::to_string(i) + "_" + std::to_string(j);

			cv::Mat img = cv::imread(filepath + imgname + ".jpg", 1);
			assert(!img.empty());
			//std::cout << filepath + imgname + ".jpg" << std::endl;

			if (img.channels() != 1) {
				cv::cvtColor(img, img, CV_RGB2GRAY);
			}

			cv::resize(img, img, img_size);

			cv::Mat mat = Extract_PDM(img);
			mat.copyTo(featureVec.row(j - 1));

			//std::cout<<i<<"  "<<j<<std::endl;

		}

		fs.open("Res\\" + std::to_string(i) + ".xml", cv::FileStorage::WRITE);
		fs << "featureVec" << featureVec;
		fs.release();
	}
}

cv::Mat myLBVLD::Extract_PDM(cv::Mat img) {
	//Extract PDM matrix from a picture

	std::vector<cv::Mat> PDMs;

	for (int m = 0; m < Block_num.height; m++) {
		for (int n = 0; n < Block_num.width; n++) {

			cv::Mat PDM = cv::Mat::zeros(Block_Size.area(), D2, CV_32FC1);
			//The first address of a pixel in a block;
			int row = m*block_stride + r;
			int col = n*block_stride + r;

			int k = 0;
			for (int r = row; r<row + Block_Size.height; r++) {
				for (int c = col; c<col + Block_Size.width; c++) {
					//Extract the PDV matrix by traversing each pixel in a block

					cv::Mat pdv = cv::Mat::zeros(1, D2, CV_32FC1);
					for (int j = 0; j<D2; j++) {
						pdv.at<float>(0, j) = img.at<uchar>(r + table_r[j][0], c + table_r[j][1]) - img.at<uchar>(r, c);
					}

					pdv.copyTo(PDM.row(k));
					k++;
					pdv.release();
				}
			}

			PDMs.push_back(PDM);

		}
	}

	assert(PDMs.size() == W.size() && W[0].rows == PDMs[0].rows&&PDMs[0].cols == V[0].rows);

	int dim = pow(2, V[0].cols);
	cv::Mat featureVec = cv::Mat::zeros(1, PDMs.size()*dim, CV_32FC1);
	for (int i = 0; i<PDMs.size(); i++) {

		cv::Mat temp = W[i].t()*PDMs[i] * V[i];
		cv::Rect rect(i*dim, 0, dim, 1);
		cv::Mat mat = Mat_Map_HashBinary(temp);

		cv::Mat vec = cv::Mat::zeros(1, dim, CV_32FC1);
		for (int i = 0; i<mat.cols; i++) {
			int temp = (int)mat.at<float>(0, i);
			vec.at<float>(0, temp)++;
		}
		vec.copyTo(featureVec(rect));
	}

	cv::Mat Vec = NormalizeVec(featureVec);

	return Vec;
}

cv::Mat myLBVLD::Mat_Map_HashBinary(cv::Mat mat) {

	for (int r = 0; r<mat.rows; r++) {
		for (int c = 0; c<mat.cols; c++) {

			mat.at<float>(r, c) = Sign_Activation(mat.at<float>(r, c));

		}
	}

	cv::Mat binary = cv::Mat::zeros(1, mat.rows, CV_32FC1);
	for (int i = 0; i<mat.rows; i++) {
		float temp = 0;
		for (int j = 0; j<mat.cols; j++) {
			temp += mat.at<float>(i, mat.cols - j - 1)*(1 << j);          //乘以2的n次方
		}
		binary.at<float>(0, i) = temp;
	}

	return binary;
}

cv::Mat myLBVLD::sign_B(cv::Mat mat) {

	for (int r = 0; r<mat.rows; r++) {
		for (int c = 0; c<mat.cols; c++) {

			mat.at<float>(r, c) = Sign_Activation(mat.at<float>(r, c));

		}
	}

	return mat;
}

cv::Mat myLBVLD::Compute_Mean_of_Matrix(cv::Mat mat) {
	//Calculate the mean of a matrix;
	//Represented by rows;

	int rows = mat.rows;
	int cols = mat.cols;

	cv::Mat mean_as_a_row;
	cv::reduce(mat, mean_as_a_row, 0, CV_REDUCE_AVG);

	assert(mean_as_a_row.rows == 1 && mean_as_a_row.cols == cols);

	cv::Mat mean;
	cv::repeat(mean_as_a_row, rows, 1, mean);

	assert(mean.rows == mat.rows&&mean.cols == mat.cols);

	return mean;
}

float myLBVLD::Sign_Activation(float i) {

	if (i> 0) {
		return 1;
	}
	else {
		return 0;
	}
}

float myLBVLD::Sigmod_Activation(float i) {

	float temp=1 / (1 + exp(-i));
	if (temp > 0.5) {
		return 1;
	}
	else {
		return 0;
	}
}

float myLBVLD::tanH_Activation(float i) {
	float temp = (1 - exp(-2 * i)) / (1 + exp(-2 * i));
	if (temp > 0) {
		return 1;
	}
	else {
		return 0;
	}
}



cv::Mat myLBVLD::NormalizeVec(cv::Mat Vec) {
	double sita = 0.001;
	double sum = 0;
	int len = Vec.cols;
	//L2-Norm and L2-Hys;
	for (int i = 0; i<len; i++) {
		sum = sum + Vec.at<float>(0, i)* Vec.at<float>(0, i);
	}
	sum = sum + sita*sita;
	sum = pow(sum, 0.5);
	for (int i = 0; i<len; i++) {
		Vec.at<float>(0, i) = Vec.at<float>(0, i) / sum;
		if (Vec.at<float>(0, i)>0.05) {
			Vec.at<float>(0, i) = 0.05;
		}
	}

	sum = 0;
	for (int i = 0; i<len; i++) {
		sum = sum + Vec.at<float>(0, i) * Vec.at<float>(0, i);
	}
	sum = sum + sita*sita;
	sum = pow(sum, 0.5);
	for (int i = 0; i<len; i++) {
		Vec.at<float>(0, i) = Vec.at<float>(0, i) / sum;
	}

	return Vec;
}

void myLBVLD::knn_classification()
{
	cv::Mat trainData, trainLabel, testData, testLabel;

	cv::Mat featureMat;
	std::string filepath = "Res\\" + std::to_string(1) + ".xml";
	cv::FileStorage fs(filepath, cv::FileStorage::READ);
	fs["featureVec"] >> featureMat;
	fs.release();

	int featureDim = featureMat.cols;
	int testnum = samplenum - trainnum;

	trainData.create(trainnum*classnum, featureDim, CV_32FC1);
	trainLabel.create(trainnum*classnum, 1, CV_32FC1);
	testData.create(testnum*classnum, featureDim, CV_32FC1);
	testLabel.create(testnum*classnum, 1, CV_32FC1);

	for (int i = 0; i<classnum; i++) {
		std::string fsname = "Res\\" + std::to_string(i + 1) + ".xml";
		featureMat.release();
		fs.open(fsname, cv::FileStorage::READ);
		fs["featureVec"] >> featureMat;
		fs.release();

		assert(featureDim == featureMat.cols);

		std::copy(featureMat.begin<float>(), featureMat.begin<float>() + featureDim*trainnum, trainData.begin<float>() + i*trainnum*featureDim);
		std::copy(featureMat.begin<float>() + featureDim*trainnum, featureMat.end<float>(), testData.begin<float>() + i*testnum*featureDim);

		for (int j = i*trainnum; j<(i + 1)*trainnum; j++) {
			trainLabel.at<float>(j, 0) = i;
		}

		for (int j = i*testnum; j < (i + 1)*testnum; j++) {
			testLabel.at<float>(j, 0) = i;
		}
		std::cout << i << std::endl;
		featureMat.release();
	}
	std::cout << "Divede Successfully" << std::endl;
	//fs.open("truelabel.xml", cv::FileStorage::WRITE);
	//fs << "data" << testLabel;
	//fs.release();

	std::vector<float> predictlabel;

	cv::Ptr < cv::ml::TrainData > tData = cv::ml::TrainData::create(trainData, cv::ml::ROW_SAMPLE, trainLabel);
	cv::Ptr < cv::ml::KNearest > knn = cv::ml::KNearest::create();
	knn->setDefaultK(1);
	knn->setIsClassifier(true);
	knn->train(tData);


	float n_sum = 0, sum_num = testData.rows;
	for (int i = 0; i<classnum; ++i)
	{
		int n = 0;
		for (int j = 0; j<testnum; ++j)
		{
			int index = i*testnum + j;
			float temp = knn->predict(testData.row(index));
			predictlabel.push_back(temp);
			if (temp == testLabel.at<float>(index, 0))
				n++;
		}
		n_sum += n;
		std::cout << i + 1 << '\t' << (float)n / testnum << std::endl;
	}

	//fs.open("predictlabel.xml", cv::FileStorage::WRITE);
	//fs << "data" << predictlabel;
	//fs.release();

	float rate = n_sum / sum_num;
	std::cout << "recognition rate:" << rate << std::endl;
	std::ofstream fout("res.txt");
	fout << rate << std::endl;   
	fout.close();
}

void myLBVLD::Kmeans_classification() {

	cv::Mat trainData, testData;

	cv::Mat featureMat;
	std::string filepath = "Res\\" + std::to_string(1) + ".xml";
	cv::FileStorage fs(filepath, cv::FileStorage::READ);
	fs["featureVec"] >> featureMat;
	fs.release();

	int featureDim = featureMat.cols;
	int testnum = samplenum - trainnum;

	trainData.create(trainnum*classnum, featureDim, CV_32FC1);
	testData.create(testnum*classnum, featureDim, CV_32FC1);

	for (int i = 0; i<classnum; i++) {
		std::string fsname = "Res\\" + std::to_string(i + 1) + ".xml";
		featureMat.release();
		fs.open(fsname, cv::FileStorage::READ);
		fs["featureVec"] >> featureMat;
		fs.release();

		assert(featureDim == featureMat.cols);

		std::copy(featureMat.begin<float>(), featureMat.begin<float>() + featureDim*trainnum, trainData.begin<float>() + i*trainnum*featureDim);
		std::copy(featureMat.begin<float>() + featureDim*trainnum, featureMat.end<float>(), testData.begin<float>() + i*testnum*featureDim);

		std::cout << i << std::endl;
		featureMat.release();
	}
	std::cout << "Divede Successfully" << std::endl;

	Kmeans kmeans(trainData, testData);
}


void myLBVLD::SVM_classification() {

	cv::Mat trainData, trainLabel, testData, testLabel;

	cv::Mat featureMat;
	std::string filepath = "Res\\" + std::to_string(1) + ".xml";
	cv::FileStorage fs(filepath, cv::FileStorage::READ);
	fs["featureVec"] >> featureMat;
	fs.release();

	int featureDim = featureMat.cols;
	int testnum = samplenum - trainnum;

	trainData.create(trainnum*classnum, featureDim, CV_32FC1);
	trainLabel.create(trainnum*classnum, 1, CV_32SC1);
	testData.create(testnum*classnum, featureDim, CV_32FC1);
	testLabel.create(testnum*classnum, 1, CV_32SC1);

	for (int i = 0; i<classnum; i++) {
		std::string fsname = "Res\\" + std::to_string(i + 1) + ".xml";
		featureMat.release();
		fs.open(fsname, cv::FileStorage::READ);
		fs["featureVec"] >> featureMat;
		fs.release();

		assert(featureDim == featureMat.cols);

		std::copy(featureMat.begin<float>(), featureMat.begin<float>() + featureDim * trainnum, trainData.begin<float>() + i * trainnum*featureDim);
		std::copy(featureMat.begin<float>() + featureDim * trainnum, featureMat.end<float>(), testData.begin<float>() + i * testnum*featureDim);

		for (int j = i * trainnum; j<(i + 1)*trainnum; j++) {
			trainLabel.at<int>(j, 0) = i;
		}

		for (int j = i * testnum; j < (i + 1)*testnum; j++) {
			testLabel.at<int>(j, 0) = i;
		}
		std::cout << i << std::endl;
		featureMat.release();
	}
	std::cout << "Divede Successfully" << std::endl;

	cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(trainData, cv::ml::ROW_SAMPLE, trainLabel);

	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();


	//	svm->setCoef0(0.0);
	//	svm->setDegree(3);
	svm->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1e6, 1e-6));
	//	svm->setGamma(0);
	svm->setKernel(cv::ml::SVM::LINEAR);   //RBF  LINEAR   SIGMOID  CHI2
	svm->setNu(0.5);
	//	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
	svm->setC(0.01); // From paper, soft classifier
	svm->setType(cv::ml::SVM::NU_SVC); // NU_SVC  C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task

									   //svm->trainAuto(tData);
	svm->train(tData);

	float n_sum = 0, sum_num = testData.rows;
	for (int i = 0; i<classnum; ++i)
	{
		int n = 0;
		for (int j = 0; j<testnum; ++j)
		{
			int index = i * testnum + j;
			float temp = svm->predict(testData.row(index));
			//std::cout << temp << std::endl;
			if (temp == testLabel.at<int>(index, 0))
				n++;
		}
		n_sum += n;
		std::cout << i + 1 << '\t' << (float)n / testnum << std::endl;
	}

	float rate = n_sum / sum_num;
	std::cout << "recognition rate:" << rate << std::endl;
	std::ofstream fout("res.txt");
	fout << rate << std::endl;   ////Save the recognition rate under the res.txt file.
	fout.close();

}

void myLBVLD::LR_classification() {
	//The effect is not very good, generally, it is not clear whether it is an OpenCV code problem or my debugging problem.
	cv::Mat trainData, trainLabel, testData, testLabel;

	cv::Mat featureMat;
	std::string filepath = "Res\\" + std::to_string(1) + ".xml";
	cv::FileStorage fs(filepath, cv::FileStorage::READ);
	fs["featureVec"] >> featureMat;
	fs.release();

	int featureDim = featureMat.cols;
	int testnum = samplenum - trainnum;

	trainData.create(trainnum*classnum, featureDim, CV_32FC1);
	trainLabel.create(trainnum*classnum, 1, CV_32FC1);
	testData.create(testnum*classnum, featureDim, CV_32FC1);
	testLabel.create(testnum*classnum, 1, CV_32FC1);

	for (int i = 0; i<classnum; i++) {
		std::string fsname = filepath + std::to_string(i + 1) + ".xml";
		featureMat.release();
		fs.open(fsname, cv::FileStorage::READ);
		fs["featureVec"] >> featureMat;
		fs.release();

		assert(featureDim == featureMat.cols);

		std::copy(featureMat.begin<float>(), featureMat.begin<float>() + featureDim * trainnum, trainData.begin<float>() + i * trainnum*featureDim);
		std::copy(featureMat.begin<float>() + featureDim * trainnum, featureMat.end<float>(), testData.begin<float>() + i * testnum*featureDim);

		for (int j = i * trainnum; j<(i + 1)*trainnum; j++) {
			trainLabel.at<float>(j, 0) = i;
		}

		for (int j = i * testnum; j < (i + 1)*testnum; j++) {
			testLabel.at<float>(j, 0) = i;
		}
		std::cout << i << std::endl;
		featureMat.release();
	}
	std::cout << "Divede Successfully" << std::endl;

	cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(trainData, cv::ml::ROW_SAMPLE, trainLabel);

	cv::Ptr<cv::ml::LogisticRegression> LR = cv::ml::LogisticRegression::create();
	LR->setIterations(1000);
	LR->setLearningRate(0.00001);
	LR->setRegularization(cv::ml::LogisticRegression::REG_L2);
	LR->setTrainMethod(cv::ml::LogisticRegression::MINI_BATCH);
	LR->setMiniBatchSize(1);

	LR->train(tData);
	//LR->save("LR.xml");

	cv::Mat reslabel;
	LR->predict(testData, reslabel);

	//cv::Ptr<cv::ml::LogisticRegression> LR = cv::ml::LogisticRegression::load("LR.xml");
	//cv::Mat reslabel;
	//LR->predict(testData, reslabel);

	float n_sum = 0, sum_num = testData.rows;
	for (int i = 0; i<classnum; ++i)
	{
		int n = 0;
		for (int j = 0; j<testnum; ++j)
		{
			int index = i * testnum + j;
			//float temp = LR->predict(testData.row(index));
			float temp = reslabel.at<int>(index, 0);
			//std::cout << temp << std::endl;
			if (temp == testLabel.at<float>(index, 0))
				n++;
		}
		n_sum += n;
		std::cout << i + 1 << '\t' << (float)n / testnum << std::endl;
	}

	float rate = n_sum / sum_num;
	std::cout << "recognition rate:" << rate << std::endl;
	std::ofstream fout("res.txt");
	fout << rate << std::endl;   
	fout.close();
}

void myLBVLD::Process() {
	//Extract_PDM();

	//DFD_Train();
	//cluster();

	//Compute_DFDVec_KMeans();

	//Compute_DFDVec_new();

	knn_classification();

	
	//SVM_classification();
	//Kmeans_classification();
	//LR_classification();

	//	Kmeans k_means(trainData,testData);

	/*
	std::vector<int> data;
	cv::FileStorage fs;
	fs.open("predictlabel.xml", cv::FileStorage::READ);
	fs["data"] >> data;
	fs.release();

	std::ofstream fout;
	fout.open("Dictlabel.txt");
	
	for (int i = 0; i < data.size(); i++) {
		fout << data.at(i) << ".";
	}

	fout.close();
	*/
}
