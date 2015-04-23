#include "auxiliary.h"
#include <cstdio>
#include <cassert>
#include <fstream>

void read_svmlight_file(const std::string& path, int N, int D, Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    std::ifstream file(path);

    X.resize(N, D);
    y.resize(N);

    std::string line;
    int sample_idx = 0;
    /* loop over samples */
    while (std::getline(file, line)) {
        assert(sample_idx < N);

        int label, feature_idx;
        double feature_value;
        int offset;
        const char* cline = line.c_str();

        /* read label */
        sscanf(cline, "%d%n", &label, &offset);
        cline += offset;
        y(sample_idx) = label;

        /* read features */
        while (sscanf(cline, " %d:%lf%n", &feature_idx, &feature_value, &offset) == 2) {
            --feature_idx; // libsvm counts from 1
            assert(feature_idx >= 0 && feature_idx < D);

            cline += offset;
            X(sample_idx, feature_idx) = feature_value;
        }

        /* go to next sample */
        ++sample_idx;
    }

    file.close();
}
