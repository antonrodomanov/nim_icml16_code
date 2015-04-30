#include <iostream>
#include <cstdio>
#include <cstring>

#include <tclap/CmdLine.h>
#include <Eigen/Dense>

#include "auxiliary.h"
#include "datasets.h"
#include "LogRegOracle.h"
#include "optim.h"
#include "Logger.h"

int main(int argc, char* argv[])
{
    /* ============================= Parse commmand-line arguments ==================================== */
    std::string method = "";
    std::string dataset = "";
    double max_epochs = 1.0;
    double n_logs_per_epoch = -1;
    double alpha = 1.0;
    double tol = 1e-9;
    double opt_allowed_time = -1;

    try {
        /* prepare parser */
        TCLAP::CmdLine cmd("Run a numerical optimiser for training logistic regression.", ' ', "0.1");

        /* specify all options */
        TCLAP::ValueArg<std::string> arg_method(
            "", "method",
            "Optimisation method (SGD, SAG, SO2)",
            true, method, "string"
        );
        TCLAP::ValueArg<std::string> arg_dataset(
            "", "dataset",
            "Dataset (a9a, mushrooms, w8a, covtype, quantum, alpha)",
            true, dataset, "string"
        );
        TCLAP::ValueArg<double> arg_max_epochs(
            "", "max_epochs",
            "Maximum number of epochs (default: 1.0)",
            false, max_epochs, "double"
        );
        TCLAP::ValueArg<double> arg_n_logs_per_epoch(
            "", "n_logs_per_epoch",
            "Number of requested logs per epoch (default: 1.0 for SGD and SAG; 10.0 for SO2)",
            false, n_logs_per_epoch, "double"
        );
        TCLAP::ValueArg<double> arg_alpha(
            "", "alpha",
            "Learning rate for SGD (default: 1.0)",
            false, alpha, "double"
        );
        TCLAP::ValueArg<double> arg_tol(
            "", "tol",
            "Gradient norm tolerance (default: 1e-9)",
            false, tol, "double"
        );
        TCLAP::ValueArg<double> arg_opt_allowed_time(
            "", "opt_allowed_time",
            "Maximal amount of time for which the optimiser is allowed to work; set -1 for no limit (default: -1)",
            false, opt_allowed_time, "double"
        );

        /* add options to parser */
        cmd.add(arg_opt_allowed_time);
        cmd.add(arg_tol);
        cmd.add(arg_n_logs_per_epoch);
        cmd.add(arg_alpha);
        cmd.add(arg_max_epochs);
        cmd.add(arg_dataset);
        cmd.add(arg_method);

        /* parse command-line string */
        cmd.parse(argc, argv);

        /* retrieve option values */
        method = arg_method.getValue();
        dataset = arg_dataset.getValue();
        max_epochs = arg_max_epochs.getValue();
        n_logs_per_epoch = arg_n_logs_per_epoch.getValue();
        if (n_logs_per_epoch == -1) {
            if (method == "SGD" || method == "SAG") {
                n_logs_per_epoch = 1.0;
            } else { // SO2
                n_logs_per_epoch = 10.0;
            }
        }
        alpha = arg_alpha.getValue();
        tol = arg_tol.getValue();
        opt_allowed_time = arg_opt_allowed_time.getValue();
    } catch (TCLAP::ArgException &e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }

    /* ============================= Load dataset ==================================== */

    Eigen::MatrixXd Z;
    Eigen::VectorXi y;

    if (dataset == "a9a") {
        fprintf(stderr, "Load a9a\n");
        load_a9a(Z, y);
    } else if (dataset == "mushrooms") {
        fprintf(stderr, "Load mushrooms\n");
        load_mushrooms(Z, y);
    } else if (dataset == "w8a") {
        fprintf(stderr, "Load w8a\n");
        load_w8a(Z, y);
    } else if (dataset == "covtype") {
        fprintf(stderr, "Load covtype\n");
        load_covtype(Z, y);
    } else if (dataset == "quantum") {
        fprintf(stderr, "Load quantum\n");
        load_quantum(Z, y);
    } else if (dataset == "alpha") {
        fprintf(stderr, "Load alpha, might take a lot of time\n");
        load_alpha(Z, y);
    } else {
        fprintf(stderr, "Unknown dataset %s\n", dataset.c_str());
        return 1;
    }

    /* ============================= Construct matrix Z ==================================== */

    Z.array().colwise() *= -y.cast<double>().array(); // multiply each sample X[i] by -y[i]

    /* ============================= Set up parameters ==================================== */

    double lambda = 1.0 / Z.rows(); // regularisation coefficient
    Eigen::VectorXd w0 = Eigen::VectorXd::Zero(Z.cols()); // starting point
    int maxiter = max_epochs * Z.rows(); // maximum number of iteration

    /* =============================== Run optimiser ======================================= */

    LogRegOracle func(Z, lambda); // prepare oracle
    Logger logger(func, n_logs_per_epoch, tol, opt_allowed_time); // prepare logger

    /* run chosen method */
    if (method == "SAG") {
        /* estimate the Lipschitz constant and step length */
        double L = 0.25 * Z.rowwise().squaredNorm().maxCoeff() + lambda;
        double alpha = 1.0 / L;

        /* print summary */
        fprintf(stderr, "Use method SAG: L=%g, alpha=%g\n", L, alpha);

        /* rum method */
        SAG(func, logger, w0, alpha, maxiter);
    } else if (method == "SGD") {
        /* print summary */
        fprintf(stderr, "Use method SGD: alpha=%g\n", alpha);

        /* run method */
        SGD(func, logger, w0, alpha, maxiter);
    } else if (method == "SO2") {
        /* print summary */
        fprintf(stderr, "Use method SO2\n");

        /* run method */
        SO2(func, logger, w0, maxiter);
    } else {
        fprintf(stderr, "Unknown method %s\n", method.c_str());
        return 1;
    }

    /* =============================== Print the trace ======================================= */

    /* construct the name of the output file */
    char out_filename[100];
    sprintf(out_filename, "output/%s_%s.dat", dataset.c_str(), method.c_str());

    /* creare output file */
    FILE* out_file;
    if (!(out_file = fopen(out_filename, "w"))) {
        fprintf(stderr, "Could not open output file '%s': %s\n", out_filename, strerror(errno));
        return 1;
    }

    /* write trace into it */
    fprintf(out_file, "%9s %9s %25s %25s\n", "epoch", "elapsed", "val", "norm_grad");
    for (int i = 0; i < int(logger.trace_epoch.size()); ++i) {
        fprintf(out_file, "%9.2f %9.3f %25.16e %25.16e\n", logger.trace_epoch[i], logger.trace_elaps[i], logger.trace_val[i], logger.trace_norm_grad[i]);
    }

    /* close output file */
    fclose(out_file);

    return 0;
}
