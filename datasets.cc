#include <fstream>
#include <cmath>
#include <cstring>

#include "datasets.h"

/* ================================================================================================================== */
/* ======================================== read_svmlight_file ====================================================== */
/* ================================================================================================================== */

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

/* ================================================================================================================== */
/* ========================================= read_pascal_file ======================================================= */
/* ================================================================================================================== */

void read_pascal_file(const std::string& dat_filename, const std::string& lab_filename, int N, int D, Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* allocate memory */
    X.resize(N, D);
    y.resize(N);

    /* read design matrix X */
    int dummy;
    FILE* file;
    if (!(file = fopen(dat_filename.c_str(), "r"))) {
        fprintf(stderr, "Could not load file '%s': %s\n", dat_filename.c_str(), strerror(errno));
        throw 1;
    }
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

    /* read labels y */
    if (!(file = fopen(lab_filename.c_str(), "r"))) {
        fprintf(stderr, "Could not load file '%s': %s\n", lab_filename.c_str(), strerror(errno));
        throw 1;
    }
    for (int i = 0; i < N; ++i) {
        dummy = fscanf(file, "%d", &y(i));
    }
    fclose(file);
}

/* ****************************************************************************************************************** */
/* ********************************************** mushrooms ********************************************************* */
/* ****************************************************************************************************************** */

void load_mushrooms(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/mushrooms/mushrooms", 8124, 112, X, y);

    /* transform y from {1, 2} to {-1, 1} */
    y = 2 * (y.array() - 1) - 1;
}

/* ****************************************************************************************************************** */
/* ************************************************* a9a ************************************************************ */
/* ****************************************************************************************************************** */

void load_a9a(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/a9a/a9a", 32561, 123, X, y);
}

/* ****************************************************************************************************************** */
/* ************************************************* w8a ************************************************************ */
/* ****************************************************************************************************************** */

void load_w8a(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/w8a/w8a", 49749 , 300, X, y);
}

/* ****************************************************************************************************************** */
/* *********************************************** covtype ********************************************************** */
/* ****************************************************************************************************************** */

void load_covtype(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* read data */
    read_svmlight_file("datasets/covtype/covtype.libsvm.binary.scale", 581012 , 54, X, y);

    /* transform y from {1, 2} to {-1, 1} */
    y = 2 * (y.array() - 1) - 1;
}

/* ****************************************************************************************************************** */
/* *********************************************** quantum ********************************************************** */
/* ****************************************************************************************************************** */

void load_quantum(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    /* set up number of samples and features */
    int N = 50000;
    int D = 78;

    /* allocate memory */
    X.resize(N, D);
    y.resize(N);

    /* read design matrix X */
    int dummy;
    FILE* file;
    file = fopen("datasets/quantum/quantum.X.scaled.dat", "r");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            dummy = fscanf(file, "%lf", &X(i, j));
        }
    }
    fclose(file);
    dummy += 0;

    /* read labels y */
    file = fopen("datasets/quantum/quantum.y.dat", "r");
    for (int i = 0; i < N; ++i) {
        dummy = fscanf(file, "%d", &y(i));
    }
    fclose(file);
}

/* ****************************************************************************************************************** */
/* ************************************************ alpha *********************************************************** */
/* ****************************************************************************************************************** */

void load_alpha(Eigen::MatrixXd& X, Eigen::VectorXi& y)
{
    read_pascal_file("datasets/alpha/alpha_train.dat", "datasets/alpha/alpha_train.lab", 500000, 500, X, y);
}
