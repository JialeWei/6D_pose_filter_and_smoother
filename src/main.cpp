#include <iostream>
#include <fstream>
#include <cassert>
#include <Eigen/Geometry>
#include <random>
#include <chrono>
#include <iomanip>
#include "../include/kalman_filter.h"
#include "../include/kalman_smoother.h"

void dataReader(std::vector<std::vector<double>> &res, const std::string &pathname) {
    std::ifstream infile;
    infile.open(pathname.data());
    assert(infile.is_open());
    std::vector<double> line;
    std::string s;
    while (getline(infile, s)) {
        std::istringstream is(s);
        double d;
        while (!is.eof()) {
            is >> d;
            line.push_back(d);
        }
        res.push_back(line);
        line.clear();
        s.clear();
    }
    infile.close();
}

Eigen::VectorXd getPose(const std::vector<double>& data_per_line){
    Eigen::VectorXd frame(12);
    for (int i = 0; i < 12; i++) {
        frame(i) = data_per_line[i];
    }
    Eigen::VectorXd pose(6);
    double x = frame(11);
    double y = frame(3);
    double z = frame(7);
    Eigen::Matrix3d rot;
    rot << frame(0), frame(1), frame(2),
            frame(4), frame(5), frame(6),
            frame(8), frame(9), frame(10);
    Eigen::Vector3d euler_angles = rot.eulerAngles(2, 1, 0);
    pose << x, y, z, euler_angles(2), euler_angles(1), euler_angles(0);
    return pose;
}


int main() {
    std::string path = "../data/poses/00.txt";

    double dt = 1.0 / 10;
    double dt2 = dt * dt / 2;
    int n = 18;
    int m = 6;

    Eigen::MatrixXd A(n, n);
    Eigen::MatrixXd C(m, n);
    Eigen::MatrixXd Q(n, n);
    Eigen::MatrixXd R(m, m);
    Eigen::MatrixXd P(n, n);

    A.setIdentity();
    A(0, 6) = dt;
    A(1, 7) = dt;
    A(2, 8) = dt;
    A(3, 9) = dt;
    A(4, 10) = dt;
    A(5, 11) = dt;
    A(6, 12) = dt;
    A(7, 13) = dt;
    A(8, 14) = dt;
    A(9, 15) = dt;
    A(10, 16) = dt;
    A(11, 17) = dt;
    A(0, 12) = dt2;
    A(1, 13) = dt2;
    A(2, 14) = dt2;
    A(3, 15) = dt2;
    A(4, 16) = dt2;
    A(5, 17) = dt2;

    C.setZero();
    C.block(0, 0, m, m) = Eigen::MatrixXd::Identity(m, m);
    Q.setIdentity() * 0.5;
    R.setIdentity() * 5.5;
    P.setIdentity();

    std::vector<std::vector<double>> data;
    dataReader(data, path);

    Eigen::VectorXd x0(n), z0(m);
    Eigen::VectorXd init_pose(6);
    init_pose = getPose(data[0]);
    x0.setZero();
    x0.head(6) = init_pose;
    z0 = init_pose;
    double t = 0;

    kalman_filter kf(x0, t, dt);
    kf.setMatrices(A, C, Q, R, P, z0);

    std::vector<Eigen::VectorXd> kf_x;
    std::vector<double> kf_t;
    std::vector< Eigen::VectorXd> measurements;
    kf_x.push_back(kf.getState());
    kf_t.push_back(kf.getTime());
    measurements.push_back(z0);



    for (int i = 1; i < data.size(); i++) {
        Eigen::VectorXd pose = getPose(data[i]);

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        const double mean = 0.0;
        const double stddev_x = 2.34;
        const double stddev_y = 2.34;
        const double stddev_z = 2.34;
        const double stddev_roll = 2.34;
        const double stddev_pitch = 2.34;
        const double stddev_yaw = 2.34;

        std::normal_distribution<double> dist_x(mean, stddev_x);
        std::normal_distribution<double> dist_y(mean, stddev_y);
        std::normal_distribution<double> dist_z(mean, stddev_z);
        std::normal_distribution<double> dist_R(mean, stddev_roll);
        std::normal_distribution<double> dist_P(mean, stddev_pitch);
        std::normal_distribution<double> dist_Y(mean, stddev_yaw);

        double x_m = pose(0) + dist_x(generator);
        double y_m = pose(1) + dist_y(generator);
        double z_m = pose(2) + dist_z(generator);
        double R_m = pose(3) + dist_R(generator);
        double P_m = pose(4) + dist_P(generator);
        double Y_m = pose(5) + dist_Y(generator);

        Eigen::VectorXd measurement(m);
        measurement << x_m, y_m, z_m, R_m, P_m, Y_m;
        kf.update(measurement);
        measurements.push_back(measurement);
        kf_x.push_back(kf.getState());
        kf_t.push_back(kf.getTime());
    }

    std::vector<Eigen::VectorXd> kf_x_ = kf.getAllStates();
    std::vector<Eigen::MatrixXd> kf_P_ = kf.getAllP();
    std::vector<Eigen::MatrixXd> kf_V_ = kf.getAllV();

    Eigen::VectorXd x_T = kf_x_.back();
    Eigen::MatrixXd V_T = kf_V_.back();
    double t_T = kf.getTime();

    kalman_smoother ks(x_T, t_T, dt);
    ks.setMatrices(A, V_T);

    for (int i = (int) data.size() - 2; i >= 0; i--) {
        Eigen::VectorXd x_in = kf_x_[i];
        Eigen::MatrixXd P_in = kf_P_[i];
        Eigen::MatrixXd V_in = kf_V_[i];
        ks.smooth(x_in, P_in, V_in);
    }

    std::vector<Eigen::VectorXd> rts_x_ = ks.getAllStates();
    std::reverse(rts_x_.begin(), rts_x_.end());
    std::cout<<kf_x.size()<<", "<<rts_x_.size()<<std::endl;
    std::cout<<data.size()<<", "<<kf_t.size()<<std::endl;


    Eigen::VectorXd measurement(12);
    double rmse_kf = 0;
    double rmse_rts = 0;

    std::ofstream out_txt_file;
    out_txt_file.open("../result/out.txt", std::ios::out | std::ios::trunc);
    out_txt_file << std::scientific;

    for (int i = 0; i < data.size(); i++) {
        Eigen::VectorXd gt_pose(6);
        gt_pose = measurements[i];
        rmse_kf += (gt_pose - C * kf_x_[i]).transpose() * (gt_pose - C * kf_x_[i]);
        rmse_rts += (gt_pose - C * rts_x_[i]).transpose() * (gt_pose - C * rts_x_[i]);
        out_txt_file << kf_t[i] << ","
                     << gt_pose(0) << ","
                     << gt_pose(1) << ","
                     << gt_pose(2) << ","
                     << gt_pose(3) << ","
                     << gt_pose(4) << ","
                     << gt_pose(5) << ","
                     << (C * kf_x_[i])(0) << ","
                     << (C * kf_x_[i])(1) << ","
                     << (C * kf_x_[i])(2) << ","
                     << (C * kf_x_[i])(3) << ","
                     << (C * kf_x_[i])(4) << ","
                     << (C * kf_x_[i])(5) << ","
                     << (C * rts_x_[i])(0) << ","
                     << (C * rts_x_[i])(1) << ","
                     << (C * rts_x_[i])(2) << ","
                     << (C * rts_x_[i])(3) << ","
                     << (C * rts_x_[i])(4) << ","
                     << (C * rts_x_[i])(5)
                     << std::endl;
    }
    out_txt_file.close();

    rmse_kf = sqrt(rmse_kf / (int) data.size());
    rmse_rts = sqrt(rmse_rts / (int) data.size());

    std::cout << "rmse_kf = " << rmse_kf << std::endl;
    std::cout << "rmse_rts = " << rmse_rts << std::endl;



//    std::vector<double> measurements = {
//            1.04202710058, 1.10726790452, 1.2913511148, 1.48485250951, 1.72825901034,
//            1.74216489744, 2.11672039768, 2.14529225112, 2.16029641405, 2.21269371128,
//            2.57709350237, 2.6682215744, 2.51641839428, 2.76034056782, 2.88131780617,
//            2.88373786518, 2.9448468727, 2.82866600131, 3.0006601946, 3.12920591669,
//            2.858361783, 2.83808170354, 2.68975330958, 2.66533185589, 2.81613499531,
//            2.81003612051, 2.88321849354, 2.69789264832, 2.4342229249, 2.23464791825,
//            2.30278776224, 2.02069770395, 1.94393985809, 1.82498398739, 1.52526230354,
//            1.86967808173, 1.18073207847, 1.10729605087, 0.916168349913, 0.678547664519,
//            0.562381751596, 0.355468474885, -0.155607486619, -0.287198661013, -0.602973173813
//    };
//
//    double dt = 1.0 / 30;
//    int n = 3;
//    int m = 1;
//
//    Eigen::MatrixXd A(n, n);
//    Eigen::MatrixXd C(m, n);
//    Eigen::MatrixXd Q(n, n);
//    Eigen::MatrixXd R(m, m);
//    Eigen::MatrixXd P(n, n);
//
//    A << 1, dt, 0, 0, 1, dt, 0, 0, 1;
//    C << 1, 0, 0;
//    Q << .05, .05, .0, .05, .05, .0, .0, .0, .0;
//    R << 10000;
//    P << .1, .1, .1, .1, 10000, 10, .1, 10, 100;
//
//    Eigen::VectorXd x0(n), z0(m);
//    double t = 0;
//    x0 << measurements[0], 0, -9.81;
//    z0 << measurements[0];
//
//    kalman_filter kf(x0, t, dt);
//    kf.setMatrices(A, C, Q, R, P, z0);
//
//    Eigen::VectorXd z(m);
//    std::vector<double> kf_x;
//    std::vector<double> kf_t;
//
//    kf_x.push_back(kf.getState().transpose()(0));
//    kf_t.push_back(kf.getTime());
//
//    for (int i = 1; i < measurements.size(); i++) {
//        //t += dt;
//        z << measurements[i];
//        //kf.predict();
//        kf.update(z);
//        kf_x.push_back(kf.getState().transpose()(0));
//        kf_t.push_back(kf.getTime());
//    }
//
//    std::vector<Eigen::VectorXd> kf_x_ = kf.getAllStates();
//    std::vector<Eigen::MatrixXd> kf_P_ = kf.getAllP();
//    std::vector<Eigen::MatrixXd> kf_V_ = kf.getAllV();
//
//    Eigen::VectorXd x_T = kf_x_.back();
//    Eigen::MatrixXd V_T = kf_V_.back();
//    double t_T = kf.getTime();
//
//    kalman_smoother ks(x_T, t_T, dt);
//    ks.setMatrices(A, V_T);
//
//    for (int i = (int) measurements.size() - 2; i >= 0; i--) {
//        Eigen::VectorXd x_in = kf_x_[i + 1];
//        Eigen::MatrixXd P_in = kf_P_[i + 1];
//        Eigen::MatrixXd V_in = kf_V_[i + 1];
//        ks.smooth(x_in, P_in, V_in);
//    }
//
//    std::vector<Eigen::VectorXd> rts_x_ = ks.getAllStates();
//    std::reverse(rts_x_.begin(), rts_x_.end());
//
//    Eigen::VectorXd measurement(m);
//    double rmse_kf = 0;
//    double rmse_rts = 0;
//
//    std::ofstream out_txt_file;
//    out_txt_file.open("../result/out.txt", std::ios::out | std::ios::trunc);
//    out_txt_file << std::fixed;
//
//    for (int i = 0; i < measurements.size(); i++) {
//        measurement << measurements[i];
//        rmse_kf += (measurement(0) - kf_x[i]) * (measurement(0) - kf_x[i]);
//        rmse_rts += (measurement(0) - rts_x_[i](0)) * (measurement(0) - rts_x_[i](0));
//
//        std::cout.setf(std::ios::fixed);
//        std::cout.width(15);
//        std::cout.setf(std::ios::showpoint);
//        std::cout.precision(3);
//
//        std::cout << "z = " << measurement<<"\t"
//        << "x_filtered = " << kf_x[i] << "\t"
//        << "x_smoothed = " << rts_x_[i](0) << std::endl;
//
//
//        out_txt_file << kf_t[i] << "," << measurement <<","<<kf_x[i]<<","<<rts_x_[i](0)<< std::endl;
//    }
//    out_txt_file.close();
//
//    rmse_kf = sqrt(rmse_kf / (int) measurements.size());
//    rmse_rts = sqrt(rmse_rts / (int) measurements.size());
//
//    std::cout << "rmse_kf = " << rmse_kf << std::endl;
//    std::cout << "rmse_rts = " << rmse_rts << std::endl;

    return 0;
}







