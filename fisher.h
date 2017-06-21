#pragma once

#include <iostream>
#include <vector>
#include <opencv2\opencv.hpp>

using cv::Mat;
using std::vector;

class Fisher 
{
public:
	Fisher(Mat & c1, Mat & c2) try :
		class1(c1.t()), class2(c2.t())
	{
		if (c1.cols != c2.cols)
			throw std::exception("Invalid input matrix!");

		Sw = Mat::zeros(c1.cols, c1.cols, CV_64FC1);
		parameters = Mat::zeros(c1.cols, 1, CV_64FC1);
		threshold = Mat::zeros(c1.cols, 1, CV_64FC1);
		mean1 = Mat::zeros(c1.cols, 1, CV_64FC1);
		mean2 = Mat::zeros(c1.cols, 1, CV_64FC1);

		w0 = 0.0;
	}
	catch (std::exception & e) {
		std::cout << e.what() << std::endl;
	}

	void train();
	int predict(Mat & data);
	vector<double> showParameters();

private:
	Mat class1;
	Mat class2;
	Mat mean1;
	Mat mean2;
	Mat Sw;
	Mat parameters;
	Mat threshold;
	double w0;
};