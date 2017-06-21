#include "fisher.h"

void Fisher::train()
{
	Mat tmpMat;

	//Calculating mean of Class1
	for (int i = 0; i < class1.cols; i++)
	{
		mean1 = mean1 + class1.col(i);
	}
	mean1 = mean1 / class1.cols;

	//Calculation mean of Class2
	for (int i = 0; i < class2.cols; i++)
	{
		mean2 = mean2 + class2.col(i);
	}
	mean2 / class2.cols;

	//Calculating threshold, namely w0
	threshold = 
		(class1.cols * mean1 + class2.cols*mean2) / (class1.cols + class2.cols);

	//Calculating covariance matrix of Class1
	Mat S1;
	S1 = Mat::zeros(class1.rows, class1.rows, CV_64FC1);
	for (int i = 0; i < class1.cols; i++)
	{
		S1 = S1 + (class1.col(i) - mean1)*(class1.col(i) - mean1).t();
	}
	S1 = S1 / (class1.cols - 1);

	//Calculating covariance matrix of Class2
	Mat S2;
	S2 = Mat::zeros(class1.rows, class1.rows, CV_64FC1);
	for (int i = 0; i < class2.cols; i++)
	{
		S2 = S2 + (class2.col(i) - mean2)*(class2.col(i) - mean2).t();
	}
	S2 = S2 / (class2.cols - 1);

	//Calculating covariance matrix of within-class
	Sw = S1 + S2;

	//Calculating parameter vector W
	parameters = Sw.inv() * (mean1 - mean2);
}

int Fisher::predict(Mat & data)
{
	Mat resultMat = parameters.t() * (data - threshold);
	w0 = ((vector<double>)(resultMat.reshape(1,1))).front();

	return w0 > 0 ? 1 : 0;
} 

vector<double> Fisher::showParameters()
{
	vector<double> tmp = (vector<double>)(parameters.reshape(1,1));
	vector<double> result;
	result.push_back(w0);
	for (auto ele : tmp)
		result.push_back(ele);

	return result;
}
