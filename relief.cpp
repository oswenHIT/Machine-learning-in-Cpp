#include "relief.h"

Relief::Relief(Mat & l, Mat & d, unsigned int num, double t) 
try:
	labels(l.t()), dataSet(d.t()),
	threshold(t), numberOfSample(num)
{
	if (l.cols != d.cols)
		throw std::exception("Invalid input data!");

	features = Mat::zeros(numberOfSample, 1, CV_64FC1);
	weights = Mat::zeros(d.cols, 1, CV_64FC1);
	weightsVector.resize(d.cols);
	for (int i = 0; i < weightsVector.size(); i++)
		weightsVector[i] = 0.0;

	vector<int> tmpVec = (vector<int>)(l.reshape(1, 1));
	for (auto ele : tmpVec)
		if (ele == 0)
			class1.push_back(ele);
		else if (ele == 1)
			class2.push_back(ele);
		else
			throw std::exception("Invalid input data!");

}
catch (const std::exception& e)
{
	cout << e.what() << endl;
}

void Relief::extractFeatures(int nf)
{
	if (nf <=0 || nf > dataSet.cols)
		return;

	std::default_random_engine e;
	std::uniform_int_distribution<> u(0, dataSet.cols - 1);

	map<double, int> nearHitData;
	map<double, int> nearMissData;
	for (int i = 0; i < numberOfSample; i++)
	{
		int rand = u(e);
		
		//Finding near-hit datapoint and near-miss datapoint
		nearHitData.clear();
		nearMissData.clear();
		if (std::find(class1.begin(), class1.end(), rand) != class1.end())
		{
			for (auto ele : class1)
			{
				if (ele != rand)
				{
					auto dot =
						calculateLength(dataSet.col(rand), dataSet.col(ele));
					nearHitData.insert(std::make_pair(dot, ele));
				}
			}

			for (auto ele : class2)
			{
				auto dot = 
					calculateLength(dataSet.col(rand), dataSet.col(ele));
				nearMissData.insert(std::make_pair(dot, ele));
			}
		}
		else
		{
			for (auto ele : class2)
			{
				if (rand != ele)
				{
					auto dot =
						calculateLength(dataSet.col(rand), dataSet.col(ele));
					nearHitData.insert(std::make_pair(dot, ele));
				}
			}

			for (auto ele : class1)
			{
				auto dot = 
					calculateLength(dataSet.col(rand), dataSet.col(ele));
				nearMissData.insert(std::make_pair(dot, ele));
			}
		}

		//Saving near-hit datapoint's index and near-miss's index
		int nearHit = (nearHitData.begin())->second;
		int nearMiss = (nearMissData.begin())->second;
		for (int i = 0; i < dataSet.rows; i++)
		{
			weightsVector[i] += calculateFeatureWeight(rand, nearHit, nearMiss, i);
		}
	}

	//Using map data sturcture to find the max-nf weights
	map<double, int, greater<int>> sortWeights;
	for (int i = 0; i < weightsVector.size(); i++)
	{
		auto ele = weightsVector[i];
		sortWeights.insert(std::make_pair(ele, i));
	}

	//Accordding to the sorted weights vector, extracting the features
	auto iter = sortWeights.begin();
	vector<Mat> matrices;
	for (int i = 0; i < nf; i++)
	{
		auto index = iter->second;
		matrices.push_back(dataSet.col(index));
		iter++;
	}
	cv::hconcat(matrices, features);
}

vector<double> Relief::showWeights()
{
	return weights;
}

Mat Relief::showReulst()
{
	return features.t();
}

//Where v1 and v2 must have the same amount of elements
double Relief::calculateLength(Mat & v1, Mat & v2)
{
	Mat diff = v1 - v2;
	auto length = diff.dot(diff);

	return length;
}

double Relief::calculateFeatureWeight(int i, int nearHit, int nearMiss, int index)
{
	double XnearHit = dataSet.col(nearHit).at<double>(index, 0);
	double XnearMiss = dataSet.col(nearMiss).at<double>(index, 0);
	double Xi = dataSet.col(i).at<double>(index, 0);
	double result = (Xi - XnearMiss)*(Xi - XnearMiss) - (Xi - XnearHit)*(Xi - XnearHit);

	return result;
}
