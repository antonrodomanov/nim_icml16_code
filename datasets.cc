#include "datasets.h"
#include "auxiliary.h"

void load_mushrooms(std::vector<std::vector<double>>& X, std::vector<int>& y)
{
    /* read data */
    read_svmlight_file("datasets/mushrooms", 8124, 112, X, y);

    /* transform y from {1, 2} to {-1, 1} */
    for (int i = 0; i < int(y.size()); ++i) {
        y[i] = (y[i] == 1) ? -1 : 1;
    }
}

void load_a9a(std::vector<std::vector<double>>& X, std::vector<int>& y)
{
    /* read data */
    read_svmlight_file("datasets/a9a", 32561, 123, X, y);
}

void load_w8a(std::vector<std::vector<double>>& X, std::vector<int>& y)
{
    /* read data */
    read_svmlight_file("datasets/w8a", 49749 , 300, X, y);
}

void load_covtype(std::vector<std::vector<double>>& X, std::vector<int>& y)
{
    /* read data */
    read_svmlight_file("datasets/covtype.libsvm.binary.scale", 581012 , 54, X, y);

    /* transform y from {1, 2} to {-1, 1} */
    for (int i = 0; i < int(y.size()); ++i) {
        y[i] = (y[i] == 1) ? -1 : 1;
    }
}
