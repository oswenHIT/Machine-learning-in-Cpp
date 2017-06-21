#include "linearegression.h"


bool LinearRegression::train(MethodType type)
{
	bool isCompleted = false;
	switch (type)
	{
	case LinearRegression::NORMAL:
		isCompleted = trainNormal();
		break;
	case LinearRegression::RANDOM:
		isCompleted = trainRandom();
		break;
	default:
		break;
	}

	return isCompleted;
}

vector<double> LinearRegression::predict(Mat & newData)
{
	Mat tmpMat;
	tmpMat = parameters * newData;
	
	return (vector<double>)(tmpMat.reshape(1, 1));
}

bool LinearRegression::trainNormal()
{
	int curIter = 0;
	while (curIter < iters)
	{
		Mat tmpMat(parameters.cols, parameters.rows, CV_64FC1);
		for (int i = 0; i < data.cols; i++)
		{
			tmpMat += (targets.col(i) - parameters * data.col(i)) * data.col(i).t();
		}

		tmpMat = tmpMat * (alpha/data.cols);
		parameters = parameters * (1-alpha*lambda) + tmpMat;

		auto error = calculateCostFunction();
		if (error < elipson)
			break;

		curIter++;
	}

	return curIter == iters ? false : true;
}

bool LinearRegression::trainRandom()
{
	std::default_random_engine e;
	std::uniform_int_distribution<> u(0, data.cols - 1);

	Mat tmpMat(parameters.cols, parameters.rows, CV_64FC1);

	int index = 0;
	for (; index < data.cols; index++)
	{
		auto randomVal = u(e);

		tmpMat = (targets.col(randomVal) - parameters * data.col(randomVal)) * data.col(randomVal).t();
		tmpMat = tmpMat * alpha;
		parameters = parameters * (1 - alpha*lambda) + tmpMat;

		auto error = calculateCostFunction();
		if (error < elipson)
			break;
	}
	
	return index == data.cols ? false : true;
}

Mat & LinearRegression::showParameters()
{
	return parameters;
}

vector<double>& LinearRegression::showErrors()
{
	return errors;
}

double LinearRegression::calculateCostFunction()
{
	Mat diff(1, 1, CV_64FC1);
	Mat tmpMat(diffs.cols, diffs.rows, CV_64FC1);
	for (int i = 0; i < data.cols; i++)
	{
		tmpMat = targets.col(i) - parameters * data.col(i);
		diff += tmpMat.t() * tmpMat;
	}

	diff = diff * 0.5;
	auto error = (vector<double>)(diff.reshape(1,1));
	errors.push_back(error.front());

	return error.front();
}