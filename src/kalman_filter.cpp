//
// Created by wjl on 12.04.21.
//

#include "../include/kalman_filter.h"

kalman_filter::kalman_filter(Eigen::VectorXd &x0, double t0, double dt) {
    x_hat_ = x0;
    dt_ = dt;
    t0_ = t0;
    t_ = t0_;
    is_initialized_ = true;
}


Eigen::VectorXd kalman_filter::getState(){
    return x_hat_;
}

std::vector< Eigen::VectorXd > kalman_filter::getAllStates(){
    return kf_x_;
}

std::vector< Eigen::MatrixXd > kalman_filter::getAllP(){
    return kf_P_;
}

std::vector< Eigen::MatrixXd > kalman_filter::getAllV(){
    return kf_V_;
}

double kalman_filter::getTime() const{
    return t_;
}

void kalman_filter::setMatrices(Eigen::MatrixXd &A, Eigen::MatrixXd &C0, Eigen::MatrixXd &Q,
                                Eigen::MatrixXd &R, Eigen::MatrixXd &P0, Eigen::VectorXd &z0) {
    A_ = A;
    C_ = C0;
    Q_ = Q;
    R_ = R;
    //P_ = P0;
    V_ = P0;


    Eigen::MatrixXd K = V_ * C_.transpose() * (C_ * V_ * C_.transpose() + R_).inverse();
    x_hat_new_ = x_hat_ + K * (z0 - C_ * x_hat_);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x_hat_.size(), x_hat_.size());
    V_ = (I - K * C_) * V_;
    x_hat_ = x_hat_new_;
    P_ = A_ * V_ * A_.transpose() + Q_;

    kf_x_.push_back(x_hat_);
    kf_P_.push_back(P_);
    kf_V_.push_back(V_);
}


void kalman_filter::update(Eigen::VectorXd &z) {
    if (is_initialized_) {
        x_hat_new_ = A_ * x_hat_;
        Eigen::MatrixXd K = P_ * C_.transpose() * (C_ * P_ * C_.transpose() + R_).inverse();
        x_hat_new_ += K * (z - C_ * x_hat_new_);
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x_hat_.size(), x_hat_.size());
        V_ = (I - K * C_) * P_;
        P_ = A_ * V_ * A_.transpose() + Q_;
        x_hat_ = x_hat_new_;
        kf_x_.push_back(x_hat_);
        kf_V_.push_back(V_);
        kf_P_.push_back(P_);
        t_ += dt_;
    }

}
