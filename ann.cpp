#include "ann.h"


ANN::ANN(Mat & l, Mat & d)
try : ratio(0.01), threshold(0.0001), func(ActFunc::SIGMOID)
{
	if (l.rows != d.rows)
		throw std::exception("Invalid input!");

	labels = l;
	dataSet = d;

	numOfInput = dataSet.cols;
	numOfHidden = static_cast<int>(std::sqrt(dataSet.cols));
	numOfOutput = l.cols;

	hiddenPars = Mat::zeros(cv::Size(numOfInput, numOfHidden), CV_64FC1);
	deltaHiddenPars = Mat::zeros(cv::Size(numOfInput, numOfHidden), CV_64FC1);
	outputPars = Mat::zeros(cv::Size(numOfHidden + 1, numOfInput), CV_64FC1);
	deltaOutputPars = Mat::zeros(cv::Size(numOfHidden + 1, numOfInput), CV_64FC1);
	double lowRange = -1 / (std::sqrt(numOfInput));
	double highRange = -lowRange;
	initParameters(lowRange, highRange);

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
	std::default_random_engine e;
	std::uniform_int_distribution<double> u(lowRange, highRange);

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

		hiddenPars = hiddenPars - ratio * (deltaHiddenPars + deltaHiddenPars);
		outputPars = outputPars - ratio * (deltaOutputPars + deltaOutputPars);

		double error = errorCalculate(func);
		errors.push_back(error);
		if (error <= threshold)
			break;
	}
}

vector<double> ANN::predict(Mat & data)
{
	calculateLayerOutputs(data);
	vector<double> rst = (vector<double>)(softmaxOutput);

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
		double sigmoid = 
			calculateSigmoid(dataSet.row(index), hiddenPars.row(i));
		hiddenOutput.at<double>(0, i) = sigmoid;
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
		double sigmoid =
			calculateSigmoid(data, hiddenPars.row(i));
		hiddenOutput.at<double>(0, i) = sigmoid;
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

double ANN::calculateSigmoid(Mat & input, Mat & pars)
{
	double dot = input.dot(pars);
	double rst = 1 / (1 + std::exp(-dot));

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

