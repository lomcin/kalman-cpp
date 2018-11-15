/**
 * Test for the KalmanFilter class with 1D projectile motion.
 *
 * @author: Hayk Martirosyan
 * @author : Lucas Oliveira Maggi (OpenCV version)
 * @date: 2014.11.15 - 2018.11.09
 */

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "kalman.hpp"


static cv::Mat calcKalm(double acc, double realDt = 1.0/30.0, int idx = 0) {

  int n = 4; // Number of states
  int m = 1; // Number of measurements

  cv::Mat A(n, n, CV_64F); // System dynamics matrix
  cv::Mat C(m, n, CV_64F); // Output matrix
  cv::Mat Q(n, n, CV_64F); // Process noise covariance
  cv::Mat R(m, m, CV_64F); // Measurement noise covariance
  cv::Mat P(n, n, CV_64F); // Estimate error covariance
  // Discrete LTI projectile motion, measuring position only
  A = (cv::Mat_<double>(n,n) << 1.0, realDt, 0.0,0.0, 1.0, realDt, 0.0, 0.0, 1.0);
  C = (cv::Mat_<double>(m,n) << 1.0, 0.0, 0.0);

  // Reasonable covariance matrices
  Q = (cv::Mat_<double>(n,n) << .05, .05, .0 , .05, .05, .0 , .0, .0, .0);
  R.at<double>(0) = 5.0;
  P = (cv::Mat_<double>(n,n) << .1, .1, .1, .1, 10000.0, 10.0, .1, 10.0, 100.0);

  std::cout << "A: \n" << A << std::endl;
/*
  std::cout << "C: \n" << C << std::endl;
  std::cout << "Q: \n" << Q << std::endl;
  std::cout << "R: \n" << R << std::endl;
  std::cout << "P: \n" << P << std::endl;
  */

  // Construct the filter
  static KalmanFilter kf[2];
  static bool initialized[2] = {0};
	if (!initialized[idx]) {
		kf[idx] = KalmanFilter(realDt, A, C, Q, R, P);

  		// Best guess of initial states
  		cv::Mat x0(n, n, CV_64F);
  		x0 = (cv::Mat_<double>(n,n) << x, 0.0, -9.81);
  		kf[idx].init(0,x0);
	}
  // Feed measurements into filter, output estimated states
  double t = 0;
  cv::Mat y(m, m, CV_64F);
  cv::Mat tmp;
  cv::transpose(kf[idx].state(),tmp);
  //std::cout << "t = " << t << ", " << "x_hat[0]: " << tmp << std::endl;
  
  
    t += realDt;
    y.at<double>(0) = x;
    kf[idx].update(y);
    cv::transpose(y,tmp);
    //std::cout << "t = " << t << ", " << "y = " << tmp;
    cv::transpose(kf[idx].state(),tmp);
    //std::cout << ", x_hat = " << tmp << std::endl;
  return tmp;
}

int main() {
  // List of noisy position measurements (y)
  std::vector<double> measurements;
  
  std::vector<double> predict;
  const double acc = 2;
  
  for(int i = 0; i < 1000; i++) {
  	measurements.push_back(i*3 +2);
  	cv::Mat tmp = calcKalm(measurements[i],1,0,acc);
  	std::cout << tmp << std::endl;
  	predict.push_back(
  		tmp.at<double>(0)
  	);

  }
  
    for(int i = 0; i < 10; i++) {
  		printf("%.4lf\n",fabs(predict[i]-measurements[i]));
  		//std::cout << measurements[i+1] << std::endl;
  	}
	return 0;
}


