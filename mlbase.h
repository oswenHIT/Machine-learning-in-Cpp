#pragma once

#include <vector>
#include <opencv2\core.hpp>

using std::vector;

class MLBase
{
public:
	//学习器训练函数
	virtual void train() = 0;

	//返回每次迭代后的训练误差
	virtual vector<double> & showErrors() = 0;
	virtual const vector<double> & showErrors() const = 0;

	//返回每次迭代后的损失函数值
	virtual vector<double> & showLossFuncVals() = 0;
	virtual const vector<double> & showLossFuncVals() const = 0;
};