//
// Created by wjl on 12.04.21.
//

#ifndef KALMAN_FILTER_AND_SMOOTHER_KALMAN_FILTER_H
#define KALMAN_FILTER_AND_SMOOTHER_KALMAN_FILTER_H


#include <iostream>
#include <Eigen/Eigen>


class kalman_filter {
public:
    kalman_filter(Eigen::VectorXd& x0, double t0, double dt);

    void setMatrices(
            Eigen::MatrixXd& A,
            Eigen::MatrixXd& C0,
            Eigen::MatrixXd& Q,
            Eigen::MatrixXd& R,
            Eigen::MatrixXd& P0,
            Eigen::VectorXd& z0
            );

    //void predict();

    void update(Eigen::VectorXd& z);

    Eigen::VectorXd getState();

    double getTime() const;

    std::vector< Eigen::VectorXd > getAllStates();

    std::vector< Eigen::MatrixXd > getAllP();

    std::vector< Eigen::MatrixXd > getAllV();


private:
    // Matrices for computation
    Eigen::MatrixXd A_, C_, Q_, R_, P_, V_;

    // Initial and current time
    double t0_, t_;

    // Discrete time step
    double dt_;

    // Is the filter initialized?
    bool is_initialized_ = false;

    // Estimated states
    Eigen::VectorXd x_hat_, x_hat_new_;

    // Save all filtered states and covariances
    std::vector< Eigen::VectorXd > kf_x_;
    std::vector< Eigen::MatrixXd > kf_P_;
    std::vector< Eigen::MatrixXd > kf_V_;

};

#endif //KALMAN_FILTER_AND_SMOOTHER_KALMAN_FILTER_H
