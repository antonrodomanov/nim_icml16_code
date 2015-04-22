#include "auxiliary.h"
#include <cstdio>
#include <cassert>
#include <fstream>

void read_svmlight_file(const std::string& path, int N, int D, std::vector<std::vector<double>>& X, std::vector<int>& y)
{
    std::ifstream file(path);

    X = std::vector<std::vector<double>>(N, std::vector<double>(D, 0.0));
    y = std::vector<int>(N, 0);

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
        y[sample_idx] = label;

        /* read features */
        while (sscanf(cline, " %d:%lf%n", &feature_idx, &feature_value, &offset) == 2) {
            --feature_idx; // libsvm counts from 1
            assert(feature_idx >= 0 && feature_idx < D);

            cline += offset;
            X[sample_idx][feature_idx] = feature_value;
        }

        /* go to next sample */
        ++sample_idx;
    }

    file.close();
}

std::vector<std::vector<double>> transform_to_z(std::vector<std::vector<double>> X, std::vector<int> y)
{
    std::vector<std::vector<double>> Z = X;

    /* multiply each sample X[i] by -y[i] */
    for (int i = 0; i < int(X.size()); ++i) {
        assert(y[i] == -1 || y[i] == 1);
        for (int j = 0; j < int(X[i].size()); ++j) {
            Z[i][j] *= -y[i];
        }
    }

    return Z;
}
