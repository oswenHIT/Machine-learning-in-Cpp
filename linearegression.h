#pragma once

#include <opencv2\core\core.hpp>
#include <vector>
#include <random>
#include <iostream>

using cv::Mat;
using std::vector;

class LinearRegression 
{
public:
	LinearRegression(Mat & dataSet, Mat & outputs, int i = 100, double e = 0.0001, double a = 0.01)
		try:
			data(dataSet.t()), targets(outputs.t()), 
			iters(i), elipson(e), alpha(a)
	{
		if (dataSet.rows != outputs.rows)
			throw std::exception("Invalid train data!");

		diffs.zeros(cv::Size(1, outputs.cols), CV_64FC1);
		parameters.zeros(cv::Size(dataSet.cols, outputs.cols), CV_64FC1);
	}
	catch(const std::exception e){
		std::cout << e.what() << std::endl;
		data = Mat::zeros(0, 0, CV_64FC1);
		targets = Mat::zeros(0, 0, CV_64FC1);
	}

	enum MethodType { NORMAL, RANDOM, REGULATION };

	Mat & showParameters();
	vector<double> & showErrors();
	bool train(MethodType type);
	vector<double> predict(Mat & newData);
	
private:
	int iters;
	double elipson;
	double alpha;
	double lambda = 2.5;
	vector<double> errors;

	Mat data;
	Mat targets;
	Mat diffs;
	Mat parameters;

	bool trainNormal();
	bool trainRandom();
	double calculateCostFunction();
};