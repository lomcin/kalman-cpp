/**
* Kalman filter implementation using Opencv. Based on the following
* introductory paper:
*
*     http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
*
* @author: Hayk Martirosyan (Eigen version)
* @author : Lucas Oliveira Maggi (OpenCV version)
* @date: 2014.11.15 - 2018.11.09
*/

#include <opencv2/opencv.hpp>

#pragma once

class KalmanFilter {

public:

  /**
  * Create a Kalman filter with the specified matrices.
  *   A - System dynamics matrix
  *   C - Output matrix
  *   Q - Process noise covariance
  *   R - Measurement noise covariance
  *   P - Estimate error covariance
  */
  KalmanFilter(
      double dt,
      cv::Mat& A,
      cv::Mat& C,
      cv::Mat& Q,
      cv::Mat& R,
      cv::Mat& P
  );

  /**
  * Create a blank estimator.
  */
  KalmanFilter();

  /**
  * Initialize the filter with initial states as zero.
  */
  void init();

  /**
  * Initialize the filter with a guess for initial states.
  */
  void init(double t0, cv::Mat& x0);

  /**
  * Update the estimated state based on measured values. The
  * time step is assumed to remain constant.
  */
  void update(cv::Mat& y, double dt = 1.0/30.0, double acc = -9.81);

  /**
  * Update the estimated state based on measured values,
  * using the given time step and dynamics matrix.
  */
  void update(cv::Mat& y, double dt, cv::Mat A);

  /**
  * Return the current state and time.
  */
  cv::Mat state() { return x_hat; };
  double time() { return t; };

private:

  // Matrices for computation
  cv::Mat A, C, Q, R, P, K, P0;

  // System dimensions
  int m, n;

  // Initial and current time
  double t0, t;

  // Discrete time step
  double dt;

  // Is the filter initialized?
  bool initialized;

  // n-size identity
  cv::Mat I;

  // Estimated states
  cv::Mat x_hat, x_hat_new;
};
