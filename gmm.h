#pragma once

#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <opencv2\core.hpp>

#include "kmeans.h"

using std::cout;
using std::endl;
using std::vector;
using std::map;
using cv::Mat;


template <typename T>
class more
{
	bool operator() (const T & t1, const T & t2)
	{
		return t1 > t2;
	}
};


class GMM 
{
public:
	GMM(Mat & data, int kinds, int i = 100, double e = 0.01);
	GMM(const GMM &) = delete;

	void train();
	int predict(Mat & dataPoint);
	const vector<double> & showLossFuncVals() const {
		return errors;
	}
	vector<double> & showLossFuncVals() { return errors; }
	const Mat & showMeans() const { return Uk; }
	Mat & showMeans() { return Uk; }

private:
	int N;
	int K;
	double elipson;
	int iters;
	double curLoss;
	double preLoss;

	Mat Nk;
	Mat PIk;
	Mat Uk;
	vector<Mat> InvCovK;
	Mat detK;
	Mat gammaZnk;
	Mat sumZnk;
	Mat dataSet;
	vector<double> errors;

	void initParameters();
	void updateGammaZnk();
	void updateSumZnk();
	void updateCovMatrix();
	void updateMeans();
	void updateNk();
	void updatePIK();
	double multiValGaussDist(Mat & dataPoint, int k);
	double calculateLossFunc();
};
