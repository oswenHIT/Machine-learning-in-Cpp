#pragma once

#include <map>
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <opencv2\opencv.hpp>

using std::vector;
using std::map;
using std::cout;
using std::endl;
using cv::Mat;

class Relief 
{
public:
	Relief(Mat & l, Mat & d, unsigned int num, double t = 0.0);
	Relief(const Relief & r) = delete;
	Relief & operator= (const Relief & rhs) = delete;

	void extractFeatures(int nf);
	vector<double> showWeights();
	Mat showReulst();

private:
	Mat dataSet;
	Mat labels;
	Mat features;
	Mat weights;
	vector<double> weightsVector;
	vector<int> class1;
	vector<int> class2;

	double threshold;
	unsigned int numberOfFeatures;
	unsigned int numberOfSample;

	double calculateLength(Mat & v1, Mat & v2);
	double calculateFeatureWeight(int i, int nearHit, int nearMiss, int index);
};

template <typename T>
class greater 
{
	bool operator() (T & t1, T & t2)
	{
		return t1 > t2;
	}
};