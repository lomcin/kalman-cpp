/**
* Implementation of KalmanFilter class.
*
* @author: Hayk Martirosyan
* @date: 2014.11.15
*/

#include <iostream>
#include <stdexcept>

#include "kalman.hpp"

KalmanFilter::KalmanFilter(
    double dt,
    cv::Mat& A,
    cv::Mat& C,
    cv::Mat& Q,
    cv::Mat& R,
    cv::Mat& P)
  : A(A), C(C), Q(Q), R(R), P0(P),
    m(C.rows), n(A.rows), dt(dt), initialized(false),
    I(n, n, CV_64F), x_hat(n,1,CV_64F), x_hat_new(n,1,CV_64F)
{
  I = cv::Mat::eye(n,n,CV_64F);
}

KalmanFilter::KalmanFilter() {}

void KalmanFilter::init(double t0, cv::Mat& x0) {
  x_hat = x0;
  P = P0;
  this->t0 = t0;
  t = t0;
  initialized = true;
  //cv::transpose(C,C);
}

void KalmanFilter::init() {
  x_hat = cv::Mat::zeros(4,1,CV_64F);
  P = P0;
  t0 = 0;
  t = t0;
  initialized = true;
}

void KalmanFilter::update(cv::Mat& y, double dt) {

  if(!initialized)
    throw std::runtime_error("Filter is not initialized!");
	
	cv::Mat tmp;
  x_hat_new = A * x_hat;//nx1
  cv::transpose(A,tmp);//nxn
  P = A*P*tmp + Q;//nxn
  cv::transpose(C,tmp);//3x1
  cv::Mat tmp4 = C*P;
  cv::Mat tmp3 = tmp4*tmp;
  cv::Mat tmp2 = (tmp3 + R).inv();
  K = P*tmp*tmp2;// 3x1
  cv::Mat tmp5 = C*x_hat_new;
  x_hat_new += K * (y - tmp5);//3x1
  P = (I - K*C)*P;//3x3
  
  x_hat = x_hat_new;//1x3

  t += dt;
  /*
  */
}

void KalmanFilter::update(cv::Mat& y, double dt, cv::Mat A) {

  this->A = A;
  this->dt = dt;
  update(y);
}

