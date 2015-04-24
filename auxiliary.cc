#include <iostream>
#include <fstream>
#include <cstdio>
#include <cassert>

#include "auxiliary.h"

void read_svmlight_file(const std::string& path, int N, int D, Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* open file */
    std::ifstream file(path);
    if (!file) {
        fprintf(stderr, "ERROR: Could not load file '%s'\n", path.c_str());
        throw 1;
    }

    /* allocate memory */
    X.resize(N, D);
    y.resize(N);

    /* loop over samples */
    std::string line;
    int sample_idx = 0;
    while (std::getline(file, line)) {
        assert(sample_idx < N); // make sure the passed number of samples is correct

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
            cline += offset;

            --feature_idx; // libsvm counts from 1
            assert(feature_idx >= 0 && feature_idx < D); // make sure the feature index is correct

            /* write this feature into the design matrix */
            X(sample_idx, feature_idx) = feature_value;
        }

        /* go to next sample */
        ++sample_idx;
    }

    /* we are done with the file, close it */
    file.close();
}
