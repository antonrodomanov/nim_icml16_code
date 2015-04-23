#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "auxiliary.h"
#include "datasets.h"
#include "LogRegOracle.h"
#include "optim.h"
#include "logger.h"

int main()
{
    std::vector<std::vector<double>> X, Z;
    std::vector<int> y;
    double lambda;

    load_a9a(X, y);
    Z = transform_to_z(X, y);

    lambda = 1.0 / X.size();

    LogRegOracle func(Z, lambda);
    std::vector<double> w0 = std::vector<double>(Z[0].size(), 0.0);

    double alpha = 1e-1;
    int maxiter = 3 * Z.size();
    //Logger logger = SGD(func, w0, alpha, maxiter);
    Logger logger = SAG(func, w0, alpha, maxiter);
    //Logger logger = SO2(func, w0, maxiter);

    printf("%9s %9s %15s %15s\n", "epoch", "elapsed", "val", "norm_grad");
    for (int i = 0; i < int(logger.trace_epoch.size()); ++i) {
        printf("%9.2f %9.2f %15.6e %15.6e\n", logger.trace_epoch[i], logger.trace_elaps[i], logger.trace_val[i], logger.trace_norm_grad[i]);
    }

    return 0;
}
