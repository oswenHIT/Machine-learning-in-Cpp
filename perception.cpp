#include "perception.h"

void Perception::train()
{
	std::default_random_engine e;
	std::uniform_int_distribution<> u(0, dataSet.cols - 1);
	for (int i = 0; i < iters; i++)
	{
		//Is data(i) in mistake dataset
		int randomValue = u(e);
		if (mistakeSet.find(randomValue) == mistakeSet.end())
			continue;

		//Updating parameters
		parameters += alpha * labels.col(randomValue) * dataSet.col(randomValue).t();
		double err = calculateErrors();
		errors.push_back(err);

		//If mistakeSet is empty, then tarining completed
		if (mistakeSet.empty())
			break;
	}
}

//dataPoint must be a n*1 matrix!
int Perception::predict(Mat & dataPoint)
{
	Mat tmp = Mat::zeros(1, 1, CV_64FC1);
	tmp = parameters * dataPoint;
	auto dv = (vector<double>)(tmp.reshape(1, 1));
	auto result = dv.front();

	return result >= 0 ? 1 : -1;
}

bool Perception::isMistake(int index)
{
	Mat tmp;
	tmp = parameters * dataSet.col(index) * labels.col(index);
	auto tmpVector = (vector<double>)(tmp.reshape(1, 1));
	double result = tmpVector.front();

	return result <= 0 ? true : false;
}

double Perception::calculateErrors()
{
	Mat tmp = Mat::zeros(1, 1, CV_64FC1);
	for (int i = 0; i < dataSet.cols; i++)
		if (isMistake(i))
		{
			if (mistakeSet.find(i) == mistakeSet.end())
				mistakeSet.insert(i);

			tmp += parameters * dataSet.col(i) * labels.col(i);
		}
		else
			if (mistakeSet.find(i) != mistakeSet.end())
				mistakeSet.erase(i);

	tmp = -1.0 * tmp;
	auto tmpVector = (vector<double>)(tmp.reshape(1, 1));

	return tmpVector.front();
}

vector<double> Perception::showParameters()
{
	return omega;
}

vector<double> Perception::showErrors()
{
	return errors;
}


