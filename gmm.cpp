#include "gmm.h"

GMM::GMM(Mat & data, int kinds, int i, double e)
try : 
	dataSet(data), K(kinds), iters(i), elipson(e)
{
	if (kinds <= 1 || i <= 1 || e <= 0.0 ||
		dataSet.cols < 1 || dataSet.rows < 1)
		throw std::exception("Invalid parameters!");

	curLoss = preLoss = 0.0;
	N = dataSet.rows;
	Nk = Mat::zeros(1, K, CV_64FC1);
	PIk = Mat::zeros(1, K, CV_64FC1);
	Uk = Mat::zeros(K, dataSet.cols, CV_64FC1);
	detK = Mat::zeros(K, dataSet.cols, CV_64FC1);
	gammaZnk = Mat::zeros(dataSet.rows, K, CV_64FC1);
	sumZnk = Mat::zeros(dataSet.rows, 1, CV_64FC1);
	for (auto & ele : InvCovK)
		ele = Mat::zeros(dataSet.cols, dataSet.cols, CV_64FC1);

	initParameters();
}
catch (const std::exception& e)
{
	cout << e.what() << endl;
}

void GMM::train()
{
	preLoss = calculateLossFunc();
	errors.push_back(preLoss);

	for (int i = 0; i < iters; i++)
	{
		//M-step of GMM
		updateNk();
		updatePIK();
		updateMeans();
		updateCovMatrix();

		//E-step of GMM
		updateGammaZnk();
		updateSumZnk();
		
		//Calculate loss function
		curLoss = calculateLossFunc();
		errors.push_back(curLoss);
		double ratio = std::abs((curLoss - preLoss) / preLoss);
		if (ratio < elipson)
			break;
		else
			preLoss = curLoss;
	}
}

int GMM::predict(Mat & dataPoint)
{
	map<double, int, more<double>> maxProb;
	for (int k = 0; k < K; k++)
	{
		double curProb = multiValGaussDist(dataPoint, k);
		maxProb.insert(std::make_pair(curProb, k));
	}

	auto max = maxProb.begin();
	int kind = max->second;

	return kind++;
}

void GMM::initParameters()
{
	//Create a Kmeans class to initialize the parameters Uk and Covk
	Kmeans k(dataSet, K);
	k.train();

	//Initializing Uk
	k.showMeans().convertTo(Uk, CV_64FC1);

	//Initializing CovK
	auto & kinds = k.showKinds();
	for (int k = 0; k < K; k++)
	{
		auto & kindSet = kinds[k];
		auto & covMatrix = InvCovK[k];
		auto & meanK = Uk.row(k);
		int Nk = kindSet.size();
		for (auto index : kindSet)
		{
			Mat diff = dataSet.row(index) - meanK;
			double * rowPtr = diff.ptr<double>(0);
			for (int row = 0; row < dataSet.cols; row++)
			{
				double * covPtr = covMatrix.ptr<double>(row);
				double * colPtr = diff.ptr<double>(0);
				for (int col = 0; col < dataSet.rows; col++)
				{
					double val = (*rowPtr) * (*colPtr);
					val = val / Nk;
					*covPtr += val;
					colPtr++;
					covPtr++;
				}
				rowPtr++;
			}
		}
	}

	//Initializing Nk
	for (int k = 0; k < K; k++)
	{
		Nk.at<double>(0, k) = 
			static_cast<double>(kinds[k].size());
	}

	//Initializing PIk
	for (int k = 0; k < K; k++)
	{
		PIk.at<double>(0, k) = Nk.at<double>(k) / N;
	}

	//Initializing Gamma Znk and sum of Znk
	updateGammaZnk();
	updateSumZnk();
}

void GMM::updateGammaZnk()
{
	for (int n = 0; n < dataSet.rows; n++)
	{
		double * gammaPtr = gammaZnk.ptr<double>(n);
		for (int k = 0; k < K; k++)
		{
			double piK = PIk.at<double>(0, k);
			*gammaPtr = piK * multiValGaussDist(dataSet.row(n), k);
			gammaPtr++;
		}
	}
}

void GMM::updateSumZnk()
{
	double * sumPtr = sumZnk.ptr<double>(0);
	for (int n = 0; n < sumZnk.cols; n++)
	{
		Mat rowGamma = gammaZnk.row(n);
		auto sum = cv::sum(rowGamma);
		*sumPtr = sum.val[0];
	}
}


double GMM::multiValGaussDist(Mat & dataPoint, int k)
{
	if (k < 0)
		throw std::exception("Invalid input!");

	double pik = PIk.at<double>(0, k);
	Mat meanK = Uk.row(k);
	double det = detK.at<double>(0, k);
	Mat invK = InvCovK[k];

	double part1 = std::pow(2 * 3.1415, dataSet.cols / 2);
	double part2 = std::sqrt(det);
	double left = 1/(part1*part2);

	Mat diff = dataPoint - meanK;
	Mat leftMul = diff*invK;
	double part3 = -0.5 * diff.dot(leftMul);
	double right = std::exp(part3);
	double rst = left * right;

	return rst;
}

double GMM::calculateLossFunc()
{
	double rst = 0.0;
	for (int n = 0; n < sumZnk.cols; n++)
	{
		double sumZn = sumZnk.at<double>(0, n);
		double tmpLn = std::log(sumZn);
		rst += tmpLn;
	}

	return rst;
}

void GMM::updateCovMatrix()
{
	for (int k = 0; k < K; k++)
	{
		InvCovK[k] = Mat::zeros(
					 InvCovK[k].rows, 
					 InvCovK[k].cols, 
					 CV_64FC1);
	}

	for (int n = 0; n < dataSet.rows; n++)
	{
		auto & curData = dataSet.row(n);
		double sZnk = sumZnk.at<double>(0, n);
		for (int k = 0; k < K; k++)
		{
			double nk = Nk.at<double>(0, k);
			double Znk = gammaZnk.at<double>(n, k);
			auto & covMatrix = InvCovK[k];
			Mat diff = curData - Uk.row(k);
			double * diffRow = diff.ptr<double>(0);
			for (int row = 0; row < dataSet.rows; row++)
			{
				double * rowPtr = covMatrix.ptr<double>(row);
				double * diffCol = diff.ptr<double>(0);
				for (int col = 0; col < dataSet.cols; col++)
				{
					double gZnk = Znk / sZnk;
					double val = gZnk * (*diffRow) * (*diffCol);
					*rowPtr += val / nk;
					diffCol++;
					rowPtr++;
				}
				diffRow++;
			}
		}
	}

	double * detPtr = detK.ptr<double>(0);
	for (int k = 0; k < K; k++)
	{
		auto & covK = InvCovK[k];
		Mat tmpCov = covK.inv();
		covK = tmpCov;
		*detPtr = cv::determinant(tmpCov);
		detPtr++;
	}
}

void GMM::updateMeans()
{
	Uk = Uk.zeros(Uk.rows, Uk.cols, CV_64FC1);
	for (int n = 0; n < dataSet.rows; n++)
	{
		Mat curData = dataSet.row(n);
		for (int k = 0; k < K; k++)
		{
			double gammaPtr = gammaZnk.at<double>(n,k);
			double nk = Nk.at<double>(0, k);
			Mat & mean = Uk.row(k);
			mean += (gammaPtr/nk) * curData;
		}
	}
}

void GMM::updateNk()
{
	Nk = Nk.zeros(Nk.rows, Nk.cols, CV_64FC1);
	for (int k = 0; k < K; k++)
	{
		Mat tmp = gammaZnk.col(k).t();
		auto sum = cv::sum(tmp);
		Nk.at<double>(0, k) = sum.val[0];
	}
}

void GMM::updatePIK()
{
	PIk = PIk.zeros(PIk.rows, PIk.cols, CV_64FC1);
	for (int k = 0; k < K; k++)
	{
		PIk.at<double>(0, k) = Nk.at<double>(0, k) / N;
	}
}

