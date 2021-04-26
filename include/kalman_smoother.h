#ifndef KALMAN_FILTER_AND_SMOOTHER_KALMAN_SMOOTHER_H
#define KALMAN_FILTER_AND_SMOOTHER_KALMAN_SMOOTHER_H

#include <iostream>
#include <Eigen/Eigen>

class kalman_smoother {
public:
    kalman_smoother(Eigen::VectorXd& x_T, double t_T, double dt);

    void setMatrices(
            Eigen::MatrixXd& A,
            Eigen::MatrixXd& V_T
    );

    void smooth(Eigen::VectorXd& x_in, Eigen::MatrixXd& P_in, Eigen::MatrixXd& V_in);


    std::vector< Eigen::VectorXd > getAllStates();


private:
    // Matrices for computation
    Eigen::MatrixXd A_, V_;

    // Initial and current time
    double t0_, t_;

    // Discrete time step
    double dt_;

    // Is the filter initialized?
    bool is_filtered_ = false;

    // Estimated states
    Eigen::VectorXd x_hat_;

    // Save all smoothed states
    std::vector< Eigen::VectorXd > rts_x_;
};

#endif //KALMAN_FILTER_AND_SMOOTHER_KALMAN_SMOOTHER_H
