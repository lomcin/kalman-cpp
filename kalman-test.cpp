/**
 * Test for the KalmanFilter class with 1D projectile motion.
 *
 * @author: Hayk Martirosyan
 * @date: 2014.11.15
 */

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "kalman.hpp"


static int calcKalm(double x, int idx = 0) {

  int n = 3; // Number of states
  int m = 1; // Number of measurements

  double dt = 1.0/30; // Time step

  cv::Mat A(n, n, CV_64F); // System dynamics matrix
  cv::Mat C(m, n, CV_64F); // Output matrix
  cv::Mat Q(n, n, CV_64F); // Process noise covariance
  cv::Mat R(m, m, CV_64F); // Measurement noise covariance
  cv::Mat P(n, n, CV_64F); // Estimate error covariance
  // Discrete LTI projectile motion, measuring position only
  A = (cv::Mat_<double>(n,n) << 1.0, dt, 0.0,0.0, 1.0, dt, 0.0, 0.0, 1.0);
  C = (cv::Mat_<double>(m,n) << 1.0, 0.0, 0.0);

  // Reasonable covariance matrices
  Q = (cv::Mat_<double>(n,n) << .05, .05, .0 , .05, .05, .0 , .0, .0, .0);
  R.at<double>(0) = 5.0;
  P = (cv::Mat_<double>(n,n) << .1, .1, .1, .1, 10000.0, 10.0, .1, 10.0, 100.0);

  std::cout << "A: \n" << A << std::endl;
  std::cout << "C: \n" << C << std::endl;
  std::cout << "Q: \n" << Q << std::endl;
  std::cout << "R: \n" << R << std::endl;
  std::cout << "P: \n" << P << std::endl;

  // Construct the filter
  static KalmanFilter kf[2];
  static bool initialized[2] = {0};
	if (!initialized[idx]) {
		kf[idx] = KalmanFilter(dt, A, C, Q, R, P);

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
  std::cout << "t = " << t << ", " << "x_hat[0]: " << tmp << std::endl;
  
  
    t += dt;
    y.at<double>(0) = x;
    kf[idx].update(y);
    cv::transpose(y,tmp);
    std::cout << "t = " << t << ", " << "y = " << tmp;
    cv::transpose(kf[idx].state(),tmp);
    std::cout << ", x_hat = " << tmp << std::endl;
  return 0;
}

int main() {
  // List of noisy position measurements (y)
  std::vector<double> measurements = {
      1.04202710058, 1.10726790452, 1.2913511148, 1.48485250951, 1.72825901034,
      1.74216489744, 2.11672039768, 2.14529225112, 2.16029641405, 2.21269371128,
      2.57709350237, 2.6682215744, 2.51641839428, 2.76034056782, 2.88131780617,
      2.88373786518, 2.9448468727, 2.82866600131, 3.0006601946, 3.12920591669,
      2.858361783, 2.83808170354, 2.68975330958, 2.66533185589, 2.81613499531,
      2.81003612051, 2.88321849354, 2.69789264832, 2.4342229249, 2.23464791825,
      2.30278776224, 2.02069770395, 1.94393985809, 1.82498398739, 1.52526230354,
      1.86967808173, 1.18073207847, 1.10729605087, 0.916168349913, 0.678547664519,
      0.562381751596, 0.355468474885, -0.155607486619, -0.287198661013, -0.602973173813
  };
  
  for(int i = 0; i < measurements.size(); i++) {
  	calcKalm(measurements[i]);
  }
	return 0;
}


