#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

void read_svmlight_file(
    const std::string& path,
    int n_samples,
    int n_features,
    std::vector<std::vector<double>>& design_matrix,
    std::vector<int>& labels
)
{
    std::ifstream file(path);

    design_matrix = std::vector<std::vector<double>>(n_samples, std::vector<double>(n_features, 0.0));
    labels = std::vector<int>(n_samples, 0);

    std::string line;
    int sample_idx = 0;
    /* loop over samples */
    while (std::getline(file, line)) {
        int label, feature_idx;
        double feature_value;
        int offset;
        const char* cline = line.c_str();

        /* read label */
        sscanf(cline, "%d%n", &label, &offset);
        cline += offset;
        labels[sample_idx] = label;

        /* read features */
        while (sscanf(cline, " %d:%lf%n", &feature_idx, &feature_value, &offset) == 2) {
            cline += offset;
            design_matrix[sample_idx][feature_idx] = feature_value;
        }

        /* go to next sample */
        ++sample_idx;
    }

    file.close();
}

int main(int argc, char** argv)
{
    std::vector<std::vector<double>> design_matrix;
    std::vector<int> labels;

    read_svmlight_file("datasets/mushrooms", 8124, 112, design_matrix, labels);

    int n_samples = 4;
    int n_features = 112;
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            printf("%g ", design_matrix[i][j]);
        }
        printf("\n");
    }

    return 0;
}
