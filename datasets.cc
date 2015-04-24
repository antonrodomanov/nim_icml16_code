#include "datasets.h"
#include "auxiliary.h"
#include <cmath>

void load_mushrooms(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/mushrooms", 8124, 112, X, y);

    /* transform y from {1, 2} to {-1, 1} */
    y = 2 * (y.array() - 1) - 1;
}

void load_a9a(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/a9a", 32561, 123, X, y);
}

void load_w8a(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/w8a", 49749 , 300, X, y);
}

void load_covtype(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/covtype.libsvm.binary.scale", 581012 , 54, X, y);

    /* transform y from {1, 2} to {-1, 1} */
    y = 2 * (y.array() - 1) - 1;
}

void load_quantum(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    int N = 50000;
    int D = 78;

    X.resize(N, D);
    y.resize(N);

    int dummy;
    FILE* file;
    file = fopen("datasets/quantum_scaled.X.dat", "r");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            dummy = fscanf(file, "%lf", &X(i, j));
        }
    }
    fclose(file);
    dummy += 0;

    file = fopen("datasets/quantum_scaled.y.dat", "r");
    for (int i = 0; i < N; ++i) {
        dummy = fscanf(file, "%d", &y(i));
    }
    fclose(file);
}

void load_alpha(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    int N = 500000;
    int D = 500;

    X.resize(N, D);
    y.resize(N);

    int dummy;
    FILE* file;
    file = fopen("datasets/alpha_train.dat", "r");
    for (int i = 0; i < N; ++i) {
        if (i % 10000 == 0) {
            fprintf(stderr, "Processed %d/%d samples (%.2f%%)\n", i, N, round(double(i) / N * 100));
        }
        for (int j = 0; j < D; ++j) {
            dummy = fscanf(file, "%lf", &X(i, j));
        }
    }
    fclose(file);
    dummy += 0;

    file = fopen("datasets/alpha_train.lab", "r");
    for (int i = 0; i < N; ++i) {
        dummy = fscanf(file, "%d", &y(i));
    }
    fclose(file);
}
