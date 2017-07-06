#include "GaussProcess.h"

void GaussProcess::setParameters(const GaussPars & par)
{
	gaussPars = par;
}

GaussPars & GaussProcess::getParameters()
{
	return gaussPars;
}

void GaussProcess::train()
{
	//Using default parameters to calculate matrix Cn
	updateKernelMatrix();
	for (int i = 0; i < iters; i++)
	{
		//Calculate the gradient of super-parameters
		calculateParameters();

		//The gradient decent algorithm
		gaussPars = gaussPars - ratio * deltaPars;
		updateKernelMatrix();

		//Calculate training errors
		double error = calculateError();
		errors.push_back(error);
		if (error - preError >= -threshold &&
			error - preError <= threshold)
			break;
	}
}

GaussDist GaussProcess::predict(Mat & dataPoints)
{
	double mean = 0.0;
	double sigma = 0.0;

	//Calculate k matrix of Gauss Process
	Mat k = Mat::zeros(1, dataSet.cols, CV_64FC1);
	double * kPtr = k.ptr<double>(0);
	for (int i = 0; i < dataSet.cols; i++)
	{
		*kPtr = calculateKernel(dataPoints, dataSet.row(i));
		kPtr++;
	}

	Mat tmp = k * CnInv;
	dist.mean = tmp.dot(dataPoints);
	dist.sigma = calculateKernel(dataPoints, dataPoints) - tmp.dot(k);

	return dist;
}

void GaussProcess::updateKernelMatrix()
{
	for (int i = 0; i < Cn.rows; i++)
	{
		double * rowPtr = Cn.ptr<double>(i);
		double * theta0Ptr = Theta0.ptr<double>(i);
		double * theta1Ptr = Theta1.ptr<double>(i);
		double * theta3Ptr = Theta3.ptr<double>(i);
		for (int j = 0; j < Cn.cols; j++)
		{
			*rowPtr++ = calculateKernel(dataSet.row(i), dataSet.row(j));
			*theta0Ptr++ = calculateTheta0(dataSet.row(i), dataSet.row(j));
			*theta1Ptr++ = calculateTheta1(dataSet.row(i), dataSet.row(j));
			*theta3Ptr++ = dataSet.row(i).dot(dataSet.row(j));
		}
	}

	CnInv = Cn.inv();
}

double GaussProcess::calculateKernel(Mat & dot1, Mat & dot2)
{
	if (dot1.size != dot2.size)
		throw std::exception("Two vectors must have same number of elements!");

	Mat diff = dot1 - dot2;
	double mul1 = diff.dot(diff);
	double mul2 = dot1.dot(dot2);
	double exp = std::exp(-0.5 * gaussPars.theta1 * mul1);
	double rst = gaussPars.theta0*exp + gaussPars.theta2 + gaussPars.theta3*mul2;

	return rst;
}

double GaussProcess::calculateTheta0(Mat & dot1, Mat & dot2)
{
	if (dot1.size != dot2.size)
		throw std::exception("Two vectors must have same number of elements!");

	Mat diff = dot1 - dot2;
	double tmp = diff.dot(diff);
	double e = -0.5*gaussPars.theta1*tmp;

	return std::exp(e);
}

double GaussProcess::calculateTheta1(Mat & dot1, Mat & dot2)
{
	if (dot1.size != dot2.size)
		throw std::exception("Two vectors must have same number of elements!");

	Mat diff = dot1 - dot2;
	double tmp = diff.dot(diff);
	double par = -0.5 * gaussPars.theta0 * tmp;
	double e = -0.5 * gaussPars.theta1 * tmp;

	return par * std::exp(e);
}

void GaussProcess::calculateParameters()
{
	Mat tmpM = -1.0 / gaussPars.alpha * Cn;
	deltaPars.alpha = calculateParameters(tmpM);
	tmpM = -1.0 / gaussPars.beta * Mat::ones(Cn.rows, Cn.cols, CV_64FC1);
	deltaPars.beta = calculateParameters(tmpM);
	deltaPars.theta0 = calculateParameters(Theta0);
	deltaPars.theta1 = calculateParameters(Theta1);
	tmpM = Mat::eye(Cn.rows, Cn.cols, CV_64FC1);
	deltaPars.theta2 = calculateParameters(tmpM);
	deltaPars.theta3 = calculateParameters(Theta3);
}

double GaussProcess::calculateParameters(Mat & m)
{
	if (m.size != CnInv.size)
		throw std::exception("Two vectors must have same size!");

	auto scaler = cv::trace(CnInv * m);
	double part1 = scaler.val[0] * 0.5;
	Mat dot = Tn.t() * CnInv;
	Mat tmp = dot * m;
	double part2 = tmp.dot(dot.t()) * -0.5;

	return part1 + part2;
}


double GaussProcess::calculateError()
{
	double det = cv::determinant(CnInv);
	double part1 = -0.5 * std::log(det);
	double part2 = -0.5 * dataSet.cols * std::log(2 * 3.1415);
	double part3 = -0.5 * (Tn.t() * CnInv).dot(Tn);

	return part1 + part2 + part3;
}


GaussPars operator- (const GaussPars & lhs, const GaussPars & rhs)
{
	return GaussPars(lhs.alpha - rhs.alpha,
		lhs.beta - rhs.beta,
		lhs.theta0 - rhs.theta0,
		lhs.theta1 - rhs.theta1,
		lhs.theta2 - rhs.theta2,
		lhs.theta3 - rhs.theta3);
}

GaussPars operator* (const GaussPars & lhs, const GaussPars & rhs)
{
	return GaussPars(lhs.alpha * rhs.alpha,
		lhs.beta * rhs.beta,
		lhs.theta0 * rhs.theta0,
		lhs.theta1 * rhs.theta1,
		lhs.theta2 * rhs.theta2,
		lhs.theta3 * rhs.theta3);
}

GaussPars operator* (double ratio, const GaussPars & rhs)
{
	return GaussPars(ratio * rhs.alpha,
		ratio * rhs.beta,
		ratio * rhs.theta0,
		ratio * rhs.theta1,
		ratio * rhs.theta2,
		ratio * rhs.theta3);
}

GaussPars operator* (const GaussPars & rhs, double ratio)
{
	return ratio * rhs;
}
