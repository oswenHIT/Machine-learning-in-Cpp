#pragma once

#include <opencv2\opencv.hpp>
#include <set>
#include <vector>
#include <iostream>
#include <random>

using cv::Mat;
using std::cout;
using std::endl;
using std::vector;
using std::set;

class Perception 
{
public:
	Perception(Mat & l, Mat & d, double a = 0.01, int i = 2000) try: 
		labels(l.t()), alpha(a), iters(i)
	{
		if (l.rows != d.rows)
			throw std::exception("Invalid input data!");

		Mat bias(l.rows, 1, CV_64FC1, cv::Scalar(1));
		cv::hconcat(bias, d, dataSet);
		dataSet = dataSet.t();

		parameters = Mat::zeros(1, dataSet.cols, CV_64FC1);
		for (int i = 0; i < l.rows; i++)
			mistakeSet.insert(i);
	}
	catch (const std::exception& e)
	{
		cout << e.what() << endl;
	}

	void train();
	int predict(Mat & dataPoint);
	vector<double> showParameters();
	vector<double> showErrors();

private:
	Mat labels;
	Mat dataSet;
	Mat parameters;

	int iters;
	double alpha;

	set<int> mistakeSet;
	vector<double> omega;
	vector<double> errors;

	bool isMistake(int index);
	double calculateErrors();
};