#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#include "auxiliary.h"
#include "datasets.h"
#include "LogRegOracle.h"
#include "optim.h"
#include "logger.h"

#include "optionparser.h"

struct Arg: public option::Arg
{
  static void printError(const char* msg1, const option::Option& opt, const char* msg2)
  {
    fprintf(stderr, "%s", msg1);
    fwrite(opt.name, opt.namelen, 1, stderr);
    fprintf(stderr, "%s", msg2);
  }

  static option::ArgStatus Unknown(const option::Option& option, bool msg)
  {
    if (msg) printError("Unknown option '", option, "'\n");
    return option::ARG_ILLEGAL;
  }

  static option::ArgStatus Required(const option::Option& option, bool msg)
  {
    if (option.arg != 0)
      return option::ARG_OK;

    if (msg) printError("Option '", option, "' requires an argument\n");
    return option::ARG_ILLEGAL;
  }

  static option::ArgStatus NonEmpty(const option::Option& option, bool msg)
  {
    if (option.arg != 0 && option.arg[0] != 0)
      return option::ARG_OK;

    if (msg) printError("Option '", option, "' requires a non-empty argument\n");
    return option::ARG_ILLEGAL;
  }

  static option::ArgStatus Numeric(const option::Option& option, bool msg)
  {
    char* endptr = 0;
    if (option.arg != 0 && strtol(option.arg, &endptr, 10)){};
    if (endptr != option.arg && *endptr == 0)
      return option::ARG_OK;

    if (msg) printError("Option '", option, "' requires a numeric argument\n");
    return option::ARG_ILLEGAL;
  }
};

enum optionIndex { UNKNOWN, HELP, DATASET, METHOD, MAX_EPOCHS };
const option::Descriptor usage[] =
{
    {UNKNOWN, 0, "" , "", option::Arg::None, "USAGE: ./main [options]\n\n"
                                             "Options:" },
    {HELP, 0, "", "help", option::Arg::None, "  --help  \tPrint usage and exit." },
    {DATASET, 0, "", "dataset", Arg::Required, "  --dataset \tDataset (a9a, mushrooms)." },
    {METHOD, 0, "", "method", Arg::Required, "  --method \tOptimisation method (SAG, SGD, SO2)." },
    {MAX_EPOCHS, 0, "", "max_epochs", Arg::Required, "  --max_epochs \tMaximum number of epochs." },
    {0,0,0,0,0,0}
};

int main(int argc, char* argv[])
{
    argc-=(argc>0); argv+=(argc>0); // skip program name argv[0] if present
    option::Stats stats(usage, argc, argv);
    option::Option options[stats.options_max], buffer[stats.buffer_max];
    option::Parser parse(usage, argc, argv, options, buffer);

    if (parse.error())
        return 1;

    if (options[HELP] || argc == 0) {
        option::printUsage(std::cout, usage);
        return 0;
    }

    if (!options[METHOD]) {
        fprintf(stderr, "Require method\n");
        return 1;
    }
    std::string method = options[METHOD].arg;

    if (!options[DATASET]) {
        fprintf(stderr, "Require dataset\n");
        return 1;
    }
    std::string dataset = options[DATASET].arg;

    if (!options[MAX_EPOCHS]) {
        fprintf(stderr, "Require max_epochs\n");
        return 1;
    }

    double max_epochs;
    sscanf(options[MAX_EPOCHS].arg, "%lf", &max_epochs);

    /* ========================================================================== */
    /* ========================================================================== */
    /* ========================================================================== */

    std::vector<std::vector<double>> Z;
    std::vector<int> y;
    double lambda;

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
    } else {
        fprintf(stderr, "Unknown dataset %s\n", dataset.c_str());
        return 1;
    }

    /* multiply each sample Z[i] by -y[i] */
    for (int i = 0; i < int(Z.size()); ++i) {
        assert(y[i] == -1 || y[i] == 1);
        for (int j = 0; j < int(Z[i].size()); ++j) {
            Z[i][j] *= -y[i];
        }
    }

    lambda = 1.0 / Z.size();

    LogRegOracle func(Z, lambda);
    std::vector<double> w0 = std::vector<double>(Z[0].size(), 0.0);

    int maxiter = max_epochs * Z.size();

    if (method == "SAG") {
        fprintf(stderr, "Use method SAG\n");

        /* choose step length */
        double L = 0.0;
        for (int i = 0; i < int(Z.size()); ++i) {
            double x2 = 0.0;
            for (int j = 0; j < int(Z[i].size()); ++j) {
                x2 += Z[i][j] * Z[i][j];
            }
            L = std::max(L, x2);
        }
        L *= 0.25;
        L += lambda; // plus reguliriser

        double alpha = 1.0 / L;
        fprintf(stderr, "SAG: L=%g, alpha=%g\n", L, alpha);
        Logger logger = SAG(func, w0, alpha, maxiter);

        printf("%9s %9s %15s %15s\n", "epoch", "elapsed", "val", "norm_grad");
        for (int i = 0; i < int(logger.trace_epoch.size()); ++i) {
            printf("%9.2f %9.2f %15.6e %15.6e\n", logger.trace_epoch[i], logger.trace_elaps[i], logger.trace_val[i], logger.trace_norm_grad[i]);
        }
    } else if (method == "SGD") {
        fprintf(stderr, "Use method SGD\n");

        double alpha = 1e-4;

        Logger logger = SGD(func, w0, alpha, maxiter);

        printf("%9s %9s %15s %15s\n", "epoch", "elapsed", "val", "norm_grad");
        for (int i = 0; i < int(logger.trace_epoch.size()); ++i) {
            printf("%9.2f %9.2f %15.6e %15.6e\n", logger.trace_epoch[i], logger.trace_elaps[i], logger.trace_val[i], logger.trace_norm_grad[i]);
        }
    } else if (method == "SO2") {
        fprintf(stderr, "Use method SO2\n");

        Logger logger = SO2(func, w0, maxiter);

        printf("%9s %9s %15s %15s\n", "epoch", "elapsed", "val", "norm_grad");
        for (int i = 0; i < int(logger.trace_epoch.size()); ++i) {
            printf("%9.2f %9.2f %15.6e %15.6e\n", logger.trace_epoch[i], logger.trace_elaps[i], logger.trace_val[i], logger.trace_norm_grad[i]);
        }
    } else {
        fprintf(stderr, "Unknown method %s\n", method.c_str());
        return 1;
    }

    return 0;
}
