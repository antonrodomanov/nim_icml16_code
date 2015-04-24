#ifndef AUXILIARY_H
#define AUXILIARY_H

#include <string>

#include <Eigen/Dense>

void read_svmlight_file(const std::string& path, int N, int D, Eigen::MatrixXd& X, Eigen::VectorXi& y);

#endif
