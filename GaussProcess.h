#pragma once

#include <opencv2\core.hpp>
#include <vector>
#include <iostream>
#include <algorithm>

#include "mlbase.h"

using std::endl;
using std::cout;
using std::vector;
using cv::Mat;

struct GaussPars 
{
	GaussPars() = default;
	GaussPars(const GaussPars &) = default;
	GaussPars & operator= (const GaussPars &) = default;
	GaussPars(double a, double b, double t0, 
		double t1, double t2, double t3) :
		alpha(a), beta(b), theta0(t0), 
		theta1(t1), theta2(t2), theta3(t3){}

	friend GaussPars operator- (const GaussPars & lhs, const GaussPars & rhs);
	friend GaussPars operator* (const GaussPars & lhs, const GaussPars & rhs);
	friend GaussPars operator* (double ratio, const GaussPars & rhs);
	friend GaussPars operator* (const GaussPars & rhs, double ratio);

	double alpha;
	double beta;
	double theta0;
	double theta1;
	double theta2;
	double theta3;
};


struct GaussDist
{
	double mean;
	double sigma;
};


class GaussProcess : public MLBase
{
public:
	GaussProcess(Mat & t, Mat & datas, double r = 0.1, int i = 100) 
	try:
		ratio(r), iters(i)
	{
		if (t.rows != datas.rows)
			throw std::exception("Invalid input data!");

		preError = 0.0;
		threshold = 0.0001;

		Tn = t;
		dataSet = datas; 3 * Cn;
		Cn = Mat::zeros(cv::Size(datas.cols, datas.cols), CV_64FC1);
		Theta0 = Mat::zeros(cv::Size(datas.cols, datas.cols), CV_64FC1);
		Theta1 = Mat::zeros(cv::Size(datas.cols, datas.cols), CV_64FC1);
		Theta3 = Mat::zeros(cv::Size(datas.cols, datas.cols), CV_64FC1);
	}
	catch (const std::exception& e)
	{
		cout << e.what() << endl;
	}

	void setParameters(const GaussPars & par);
	GaussPars & getParameters();
	void train() override;
	GaussDist predict(Mat & dataPoints);
	vector<double> & showErrors() override { return errors; }
	const vector<double> & showErrors() const override { return errors; }

private:
	//Transit matrixs
	Mat Tn;
	Mat Cn;
	Mat CnInv;
	Mat Theta0;
	Mat Theta1;
	Mat Theta3;
	Mat dataSet;

	//Super parameters of Gauss Process
	GaussPars gaussPars;
	GaussPars deltaPars;
	GaussDist dist;
	double ratio;
	double threshold;
	double preError;
	int iters;
	vector<double> errors;

	//Private calculation functions
	void updateKernelMatrix();
	double calculateKernel(Mat & dot1, Mat & dot2);
	double calculateTheta0(Mat & dot1, Mat & dot2);
	double calculateTheta1(Mat & dot1, Mat & dots);
	void calculateParameters();
	double calculateParameters(Mat & m);
	double calculateError();
};
