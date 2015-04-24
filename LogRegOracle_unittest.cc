#include "LogRegOracle.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>

TEST(NSamplesTest, Basic) {
    Eigen::MatrixXd Z;
    Eigen::VectorXd w;
    double lambda;

    Z.resize(3, 2);
    Z << 0.1, -0.01,
         -0.01, -0.1,
         0.3, 0.1;
    w.resize(2);
    w << 0.1, -0.4;
    lambda = 0.1;

    LogRegOracle func(Z, lambda);

    EXPECT_EQ(func.n_samples(), 3);
}

TEST(SingleValTest, Basic) {
    Eigen::MatrixXd Z;
    Eigen::VectorXd w;
    double lambda;

    Z.resize(3, 2);
    Z << 0.1, -0.01,
         -0.01, -0.1,
         0.3, 0.1;
    w.resize(2);
    w << 0.1, -0.4;
    lambda = 0.1;

    LogRegOracle func(Z, lambda);

    EXPECT_NEAR(func.single_val(w, 1), 0.721337293512, 1e-5);
}

TEST(SingleValTest, Overflow) {
    Eigen::MatrixXd Z;
    Eigen::VectorXd w;
    double lambda;

    Z.resize(3, 2);
    Z << 0.1, -0.01,
         -0.01, -0.1,
         0.3, 0.1;
    w.resize(2);
    w << 1e5, 1e5;
    lambda = 0.1;

    LogRegOracle func(Z, lambda);

    EXPECT_NEAR(func.single_val(w, 2), 1000040000.0, 1e-5);
}

TEST(SingleGradTest, Basic) {
    Eigen::MatrixXd Z;
    Eigen::VectorXd w;
    double lambda;

    Z.resize(3, 2);
    Z << 0.1, -0.01,
         -0.01, -0.1,
         0.3, 0.1;
    w.resize(2);
    w << 0.1, -0.4;
    lambda = 0.1;

    LogRegOracle func(Z, lambda);

    Eigen::VectorXd gi = func.single_grad(w, 1);
    EXPECT_NEAR(gi(0), 0.00490251, 1e-5);
    EXPECT_NEAR(gi(1), -0.09097488, 1e-5);
}

TEST(FullValTest, Basic) {
    Eigen::MatrixXd Z;
    Eigen::VectorXd w;
    double lambda;

    Z.resize(3, 2);
    Z << 0.1, -0.01,
         -0.01, -0.1,
         0.3, 0.1;
    w.resize(2);
    w << 0.1, -0.4;
    lambda = 0.1;

    LogRegOracle func(Z, lambda);

    EXPECT_NEAR(func.full_val(w), 0.70888955146, 1e-5);
}

TEST(FullValTest, Overflow) {
    Eigen::MatrixXd Z;
    Eigen::VectorXd w;
    double lambda;

    Z.resize(3, 2);
    Z << 0.1, -0.01,
         -0.01, -0.1,
         0.3, 0.1;
    w.resize(2);
    w << 1e5, 1e5;
    lambda = 0.1;

    LogRegOracle func(Z, lambda);

    EXPECT_NEAR(func.full_val(w), 1000016333.333333, 1e-5);
}

TEST(FullGradTest, Basic) {
    Eigen::MatrixXd Z;
    Eigen::VectorXd w;
    double lambda;

    Z.resize(3, 2);
    Z << 0.1, -0.01,
         -0.01, -0.1,
         0.3, 0.1;
    w.resize(2);
    w << 0.1, -0.4;
    lambda = 0.1;

    LogRegOracle func(Z, lambda);

    Eigen::VectorXd g = func.full_grad(w);
    EXPECT_NEAR(g(0), 0.07483417, 1e-5);
    EXPECT_NEAR(g(1), -0.04208662, 1e-5);
}
