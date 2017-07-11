#pragma once

#include <vector>
#include <opencv2\core.hpp>

using std::vector;

class MLBase
{
public:
	//ѧϰ��ѵ������
	virtual void train() = 0;

	//����ÿ�ε������ѵ�����
	virtual vector<double> & showErrors() = 0;
	virtual const vector<double> & showErrors() const = 0;

	//����ÿ�ε��������ʧ����ֵ
	virtual vector<double> & showLossFuncVals() = 0;
	virtual const vector<double> & showLossFuncVals() const = 0;
};