#pragma once

#include <opencv2\core.hpp>
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>


using cv::Mat;
using std::endl;
using std::cout;
using std::vector;


struct Parameters
{
	Parameters(double r, double t, double l1, double l2, 
		int input, int hidden, int output):
		ratio(r), threshold(t), numOfHidden(hidden),
		lambda1(l1), lambda2(l2){}

	double ratio;
	double lambda1;
	double lambda2;
	double threshold;
	int numOfHidden;
};

class ANN 
{
public:
	enum ActFunc
	{
		SIGMOID, LINEAR, SOFTMAX
	};

public:
	ANN() = default;
	ANN(const ANN & rhs) = delete;
	ANN(Mat & l, Mat & d);
	~ANN() { ; }

	void setRatio(double r) 
	{ 
		ratio = r; 
	}
	void setThresh(double t) 
	{ 
		threshold = t; 
	}
	void setNumOfHidden(int n) 
	{ 
		if (n > 0)
			numOfHidden = n; 
	}
	void setActivationFunction(ActFunc f)
	{
		func = f;
	}
	void setParameters(Parameters pars)
	{
		ratio = pars.ratio;
		lambda1 = pars.lambda1;
		lambda2 = pars.lambda2;
		threshold = pars.threshold;
		numOfHidden = pars.numOfHidden;
	}

	void train();
	vector<double> predict(Mat & data);
	vector<double> showErrors() 
	{ 
		return errors; 
	}

private:
	Mat labels;
	Mat dataSet;
	Mat hiddenPars;
	Mat deltaHiddenPars;
	Mat outputPars;
	Mat deltaOutputPars;
	Mat hiddenOutput;
	Mat output;
	Mat sigmoidOutput;
	Mat softmaxOutput;
	Mat outputDiff;

	vector<double> errors;
	double ratio;
	double lambda1;
	double lambda2;
	double threshold;
	int numOfInput;
	int numOfHidden;
	int numOfOutput;
	ActFunc func;

	void initParameters(double lowRange, double highRange);
	void calculateParameters(int index);
	void calculateLayerOutputs(int index);
	void calculateLayerOutputs(Mat & data);
	double errorCalculate(ActFunc func);
	double calculateSigmoid(Mat & input, Mat & pars);
	void calculateSoftmax(Mat & data);
	void calculateSigmoid(Mat & data);
	double squareError();
	double sigmoidError();
	double softmaxError();
};