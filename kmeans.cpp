#include "kmeans.h"

void Kmeans::train()
{
	//随机初始化K个类别的均值向量
	std::default_random_engine e;
	std::uniform_int_distribution<int> u(0, dataSet.rows - 1);
	for (int i = 0; i < K; i++)
	{
		int randomVal = u(e);
		double * meanPtr = preMeans.ptr<double>(i);
		double * distPtr = dataSet.ptr<double>(randomVal);
		for (int j = 0; j < dataSet.cols; j++)
			*meanPtr++ = *distPtr++;
	}

	for (int i = 0; i < iters; i++)
	{
		//为每一个数据点分配类别
		for (int j = 0; j < dataSet.rows; j++)
		{
			minDist.clear();
			for (int k = 0; k < K; k++)
			{
				double dist = 
					calculateDist(dataSet.row(j), preMeans.row(k));
				minDist.insert(std::make_pair(dist, k));
			}

			//根据最小距离的数据索引值，将改数据添加到相应的类别容器中
			auto min = minDist.begin();
			kinds[min->second].push_back(j);
		}

		//根据更新的索引值计算当前的K个类的均值向量，并进行误差计算
		updateKMeans();
		double error = calculateError();
		errors.push_back(error);
		if (error < threshold)
			break;

		//清空类别容器
		for (auto ele : kinds)
			ele.clear();
	}
}

int Kmeans::predict(Mat & data)
{
	int kind = 0;
	double maxDist = 0.0;
	for (int k = 0; k < K; k++)
	{
		double curDist = calculateDist(data, curMeans.row(k));
		if (curDist > maxDist)
		{
			kind = k;
			maxDist = curDist;
		}
	}

	return kind + 1;
}

vector<double>& Kmeans::showErrors()
{
	return errors;
}

double Kmeans::calculateDist(Mat & rhs, Mat & lhs)
{
	if (rhs.rows != lhs.rows || rhs.cols != lhs.cols)
		throw std::exception("Two vectors must have the same size!");

	Mat diff = rhs - lhs;
	double dot = diff.dot(diff);

	return dot;
}

void Kmeans::updateKMeans()
{
	for (int k = 0; k < K; k++)
	{
		Mat curMean = curMean.row(k);
		for (auto ele : kinds[k])
		{
			curMean = curMean + dataSet.row(ele);
		}
		curMean = curMean / kinds[k].size();
	}
}

double Kmeans::calculateError()
{
	Mat diff = curMeans - preMeans;
	double error = diff.dot(diff);
	error = std::sqrt(error);

	return error;
}
