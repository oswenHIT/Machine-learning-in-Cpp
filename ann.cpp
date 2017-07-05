#include "ann.h"


ANN::ANN(Mat & l, Mat & d)
try : ratio(0.01), lambda1(0.5), lambda2(0.5), 
	  threshold(0.0001), func(ActFunc::SIGMOID)
{
	if (l.rows != d.rows)
		throw std::exception("Invalid input!");

	labels = l;
	dataSet = d;

	//默认将隐含层选为输入特征数目的0.5次方
	numOfInput = dataSet.cols;
	numOfHidden = static_cast<int>(std::sqrt(dataSet.cols));
	numOfOutput = l.cols;
	hiddenPars = Mat::zeros(cv::Size(numOfInput, numOfHidden), CV_64FC1);
	deltaHiddenPars = Mat::zeros(cv::Size(numOfInput, numOfHidden), CV_64FC1);
	outputPars = Mat::zeros(cv::Size(numOfHidden + 1, numOfInput), CV_64FC1);
	deltaOutputPars = Mat::zeros(cv::Size(numOfHidden + 1, numOfInput), CV_64FC1);

	//随机初始化隐层和输出层参数矩阵
	double lowRange = -1 / (std::sqrt(numOfInput));
	double highRange = -lowRange;
	initParameters(lowRange, highRange);

	//将各个输出矩阵初始化为0
	hiddenOutput = Mat::zeros(cv::Size(numOfHidden + 1, 1), CV_64FC1);
	output = Mat::zeros(cv::Size(numOfOutput, 1), CV_64FC1);
	softmaxOutput = Mat::zeros(cv::Size(numOfOutput, 1), CV_64FC1);
	sigmoidOutput = Mat::zeros(cv::Size(numOfOutput, 1), CV_64FC1);
	outputDiff = Mat::zeros(cv::Size(numOfOutput, 1), CV_64FC1);
}
catch (const std::exception& e)
{
	cout << e.what() << endl;
}

void ANN::initParameters(double lowRange, double highRange)
{
	//初始化随机数发生器
	std::default_random_engine e;
	std::uniform_int_distribution<double> u(lowRange, highRange);

	//初始化隐含层参数矩阵
	int rows = hiddenPars.rows;
	int cols = hiddenPars.cols;
	for (int i = 0; i < rows; i++)
	{
		double * data = hiddenPars.ptr<double>(i);
		for (int j = 0; j < cols; j++)
		{
			double random = u(e);
			*data++ = random;
		}
	}

	//初始化输出层参数矩阵
	rows = outputPars.rows;
	cols = outputPars.cols;
	for (int i = 0; i < rows; i++)
	{
		double * data = outputPars.ptr<double>(i);
		for (int j = 0; j < cols; j++)
		{
			double random = u(e);
			*data++ = random;
		}
	}
}

void ANN::train()
{
	for (int i = 0; i < dataSet.rows; i++)
	{
		calculateLayerOutputs(i);

		switch (func)
		{
		case ANN::SIGMOID:
			outputDiff = sigmoidOutput - labels.row(i);
			break;
		case ANN::LINEAR:
			outputDiff = output - labels.row(i);
			break;
		case ANN::SOFTMAX:
			outputDiff = softmaxOutput - labels.row(i);
			break;
		default:
			break;
		}

		calculateParameters(i);

		//梯度下降算法，误差函数为带有L2正则化项的误差函数
		hiddenPars = hiddenPars -
			ratio * (deltaHiddenPars + lambda1 * deltaHiddenPars);
		outputPars = outputPars - 
			ratio * (deltaOutputPars + lambda2 * deltaOutputPars);

		double error = errorCalculate(func);
		errors.push_back(error);
		if (error <= threshold)
			break;
	}
}

vector<double> ANN::predict(Mat & data)
{
	calculateLayerOutputs(data);

	vector<double> rst;
	switch (func)
	{
	case ANN::SIGMOID:
		rst = (vector<double>)(sigmoidOutput.reshape(1, 1));
		break;
	case ANN::LINEAR:
		rst = (vector<double>)(output.reshape(1, 1));
		break;
	case ANN::SOFTMAX:
		rst = (vector<double>)(softmaxOutput.reshape(1, 1));
		break;
	default:
		break;
	}

	return rst;
}

void ANN::calculateParameters(int index)
{
	//Calculating differeciation of output layer's parameters
	double * diffPtr = outputDiff.ptr<double>(0);
	for (int k = 0; k < outputDiff.cols; k++)
	{
		double * tmpPtr = deltaOutputPars.ptr<double>(k);
		for (int j = 0; j < hiddenOutput.cols; j++)
		{
			*tmpPtr = *diffPtr * hiddenOutput.at<double>(0, j);
			tmpPtr++;
		}

		diffPtr++;
	}

	//Calculating differeciation of hidden layer's parameters
	for (int j = 1; j < hiddenOutput.cols; j++)
	{
		double sum = 0.0;
		for (int k = 0; k < outputDiff.cols; k++)
		{
			double deltaK = outputDiff.at<double>(0, k);
			double Wkj = outputPars.at<double>(k, j);
			double mul = deltaK * Wkj;
			sum += mul;
		}

		for (int i = 0; i < dataSet.cols; i++)
		{
			double Zj = hiddenOutput.at<double>(0, j);
			Zj = 1 - Zj * Zj;
			double Xi = dataSet.row(index).at<double>(i);
			double Wji = Xi * Zj * sum;
			deltaHiddenPars.at<double>(j, i) = Wji;
		}
	}
}

void ANN::calculateLayerOutputs(int index)
{
	//Calculating hidden layer's outputs
	hiddenOutput.at<double>(0, 0) = 1.0;
	for (int i = 1; i < hiddenOutput.cols; i++)
	{
		double tanh = 
			calculateTanh(dataSet.row(index), hiddenPars.row(i));
		hiddenOutput.at<double>(0, i) = tanh;
	}

	//Calculating output layer
	for (int i = 0; i < output.cols; i++)
	{
		double rst = output.row(i).dot(hiddenOutput);
		output.at<double>(0, i) = rst;
	}

	//Calculating softmax or sigmoid
	switch (func)
	{
	case ANN::SIGMOID:
		calculateSigmoid(output);
		break;
	case ANN::SOFTMAX:
		calculateSoftmax(output);
		break;
	default:
		break;
	}
}

void ANN::calculateLayerOutputs(Mat & data)
{
	//Calculating hidden layer's outputs
	hiddenOutput.at<double>(0, 0) = 1.0;
	for (int i = 1; i < hiddenOutput.cols; i++)
	{
		double tanh =
			calculateTanh(data, hiddenPars.row(i));
		hiddenOutput.at<double>(0, i) = tanh;
	}

	//Calculating output layer
	for (int i = 0; i < output.cols; i++)
	{
		double rst = output.row(i).dot(hiddenOutput);
		output.at<double>(0, i) = rst;
	}

	//Calculating softmax or sigmoid
	switch (func)
	{
	case ANN::SIGMOID:
		calculateSigmoid(output);
		break;
	case ANN::SOFTMAX:
		calculateSoftmax(output);
		break;
	default:
		break;
	}
}

double ANN::errorCalculate(ActFunc func)
{
	double error = 0.0;
	switch (func)
	{
	case ANN::SIGMOID:
		error = sigmoidError();
		break;
	case ANN::LINEAR:
		error = squareError();
		break;
	case ANN::SOFTMAX:
		error = softmaxError();
		break;
	default:
		break;
	}
	
	return error;
}

double ANN::squareError()
{
	double sumAll = 0.0;
	for (int i = 0; i < dataSet.rows; i++)
	{
		calculateLayerOutputs(i);

		outputDiff = output - labels.row(i);
		double sum = 0.0;
		for (int k = 0; k < outputDiff.cols; k++)
		{
			double tmp = outputDiff.at<double>(0, k);
			sum += tmp * tmp;
		}
		sum /= dataSet.rows;

		sumAll += sum;
	}

	return sumAll;
}

double ANN::sigmoidError()
{
	double sumAll = 0.0;
	for (int i = 0; i < dataSet.rows; i++)
	{
		calculateLayerOutputs(i);

		double sum = 0.0;
		for (int k = 0; k < sigmoidOutput.cols; k++)
		{
			double Yk = softmaxOutput.at<double>(0, k);
			double Tk = labels.row(i).at<double>(0, k);
			sum += Tk*std::log(Yk) + (1 - Tk)*std::log(1 - Yk);
		}
		sum /= dataSet.rows;

		sumAll += sum;
	}

	return sumAll;
}

double ANN::softmaxError()
{
	double sumAll = 0.0;
	for (int i = 0; i < dataSet.rows; i++)
	{
		calculateLayerOutputs(i);

		double sum = 0.0;
		for (int k = 0; k < softmaxOutput.cols; k++)
		{
			double Yk = softmaxOutput.at<double>(0, k);
			double Tk = labels.row(i).at<double>(0, k);
			sum += Tk*std::log(Yk);
		}
		sum /= dataSet.rows;

		sumAll += sum;
	}

	return sumAll;

	return 0.0;
}

double ANN::calculateTanh(Mat & input, Mat & pars)
{
	double dot = input.dot(pars);
	double numerator = std::exp(dot) - std::exp(-dot);
	double dominator = std::exp(dot) + std::exp(-dot);
	double rst = numerator / dominator;

	return rst;
}

double ANN::calculateSigmoid(Mat & input, Mat & pars)
{
	double dot = input.dot(pars);
	double dominator = 1.0 + std::exp(-dot);
	double rst = 1 / dominator;

	return rst;
}

void ANN::calculateSigmoid(Mat & data)
{
	double * ptr = data.ptr<double>(0);
	double * sigmoidPtr = sigmoidOutput.ptr<double>(0);
	for (int i = 0; i < softmaxOutput.cols; i++)
	{
		double e = std::exp(-*ptr);
		e = 1 / (1 + e);
		*sigmoidPtr = e;
		sigmoidPtr++;
		ptr++;
	}
}

void ANN::calculateSoftmax(Mat & data)
{
	double * ptr = data.ptr<double>(0);
	double * softPtr = softmaxOutput.ptr<double>(0);
	for (int i = 0; i < softmaxOutput.cols; i++)
	{
		double e = std::exp(*ptr);
		*softPtr = e;
		softPtr++;
		ptr++;
	}

	auto tmp = cv::sum(softmaxOutput);
	double sum = tmp.val[0];
	for (int i = 0; i < softmaxOutput.cols; i++)
	{
		*softPtr = *softPtr / sum;
		softPtr++;
		ptr++;
	}
}

