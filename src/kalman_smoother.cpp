#include "../include/kalman_smoother.h"

kalman_smoother::kalman_smoother(Eigen::VectorXd &x_T, double t_T, double dt) {
    x_hat_ = x_T;
    rts_x_.push_back(x_hat_);
    dt_ = dt;
    t0_ = t_T;
    t_ = t0_;
    is_filtered_ = true;
}

std::vector< Eigen::VectorXd > kalman_smoother::getAllStates(){
    return rts_x_;
}

void kalman_smoother::setMatrices(Eigen::MatrixXd &A, Eigen::MatrixXd& V_T) {
    A_ = A;
    V_ = V_T;
}

void kalman_smoother::smooth(Eigen::VectorXd &x_in, Eigen::MatrixXd& P_in, Eigen::MatrixXd& V_in) {
    if (is_filtered_){
        Eigen::MatrixXd G = V_in * A_.transpose() * P_in.inverse();
        x_hat_ = x_in + G * (x_hat_ - A_ * x_in);
        V_ = V_in + G * (V_ - P_in) * G.transpose();
        rts_x_.push_back(x_hat_);
        t_ -= dt_;
    }

}