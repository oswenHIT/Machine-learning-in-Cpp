#pragma once

#include <vector>
#include <map>
#include <utility>
#include <iostream>
#include <opencv2\core.hpp>
#include <algorithm>
#include <random>

#include "mlbase.h"

using cv::Mat;
using std::vector;
using std::map;
using std::pair;
using std::cout;
using std::endl;


class Kmeans : public MLBase
{
public:
	Kmeans(Mat & datas, int k, double t = 0.005, int i = 100)
	try : dataSet(datas), K(k), threshold(t), iters(i)
	{
		if (K <= 1)
			throw std::exception("Kinds must greater than zero!");

		curMeans = Mat::zeros(K, dataSet.cols, CV_64FC1);
		preMeans = Mat::zeros(K, dataSet.cols, CV_64FC1);
		kinds.resize(K);
	}
	catch (const std::exception & e)
	{
		cout << e.what() << endl;
	}

	void train() override;
	int predict(Mat & data);
	void setKinds(int k)
	{
		if (k >= 1)
			K = k;
	}
	vector<double> & showErrors() override;
	const vector<double> & showErrors() const override;
	Mat & showMeans() { return curMeans; }
	vector<vector<int>> & showKinds() { return kinds; }

private:
	Mat dataSet;
	Mat curMeans;
	Mat preMeans;

	double threshold;
	int iters;
	int K;
	vector<double> errors;
	vector<vector<int>> kinds;
	map<double, int> minDist;

	double calculateDist(Mat & rhs, Mat & lhs);
	void updateKMeans();
	double calculateError();
};
