#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "auxiliary.h"

int main(int argc, char** argv)
{
    std::vector<std::vector<double>> X, Z;
    std::vector<int> y;

    read_svmlight_file("datasets/mushrooms", 8124, 112, X, y);

    /* transform y from {1, 2} to {-1, 1} */
    for (int i = 0; i < int(y.size()); ++i) {
        y[i] = (y[i] == 1) ? -1 : 1;
    }

    Z = transform_to_z(X, y);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < int(Z[i].size()); ++j) {
            printf("%g ", Z[i][j]);
        }
        printf("\n");
    }

    return 0;
}
