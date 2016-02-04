#include <iostream>
#include <vector>
#include <cstdio>
#include <ctime>

#include "Logger.h"
#include "special.h"

Logger::Logger(const LogRegOracle& func, double n_logs_per_epoch, double tol, double opt_allowed_time) :
    func(func),
    how_often(func.n_samples() / n_logs_per_epoch),
    n_calls(0),
    t_start(clock()),
    mainten_time(0.0),
    tol(tol),
    opt_allowed_time(opt_allowed_time)
{}

bool Logger::log(const Eigen::VectorXd& w, size_t n_full_calls)
{
    bool terminate = false; // don't terminate unless needed

    if (n_full_calls > 0) { /* this is a non-incremental optimiser, its each call equals one epoch */
        n_calls = n_full_calls * func.n_samples();
    }

    if (n_calls % how_often == 0) { /* perform actual logging only every `how_often` calls */
        /* remember the time when the maintenance starts */
        clock_t mainten_start = clock();

        /* calculate the number of epoch and optimiser's time */
        double epoch = double(n_calls) / func.n_samples(); // current epoch
        double tot_elaps = double(clock() - t_start) / CLOCKS_PER_SEC; // total time elapsed since optimiser was run
        double opt_elaps = tot_elaps - mainten_time; // `tot_elaps` equals optimiser's time + maintenance time, so
                                                     // subtract maintenance time to get optimiser's time
        /* always start `opt_elaps` at zero (first call is for free) */
        if (n_calls == 0) {
            opt_elaps = 0.0;
        }

        /* calculate function value and composite gradient norm */
        double f = func.full_val(w) + func.lambda1 * w.lpNorm<1>(); // smooth part + non-smooth
        double norm_g = (w - func.prox1(w - func.full_grad(w), 1)).lpNorm<Eigen::Infinity>(); // composite gradient with step length 1

        /* print calculated values */
        if (n_calls == 0) { /* print table header when it's the first call */
            fprintf(stderr, "%9s %9s %9s %25s %15s\n", "epoch", "opt_elaps", "tot_elaps", "f", "norm_g");
        }
        fprintf(stderr, "%9.2f %9.2f %9.2f %25.16f %15.6e\n", epoch, opt_elaps, tot_elaps, f, norm_g);

        /* append calculated values to trace */
        trace_epoch.push_back(epoch);
        trace_elaps.push_back(opt_elaps);
        trace_val.push_back(f);
        trace_norm_grad.push_back(norm_g);

        /* tell the optimiser to terminate if needed */
        if (norm_g < tol) {
            fprintf(stderr, "Reached gradient tolerance tol=%g\n", tol);
            terminate = true;
        }
        if (opt_allowed_time != -1 && opt_elaps > opt_allowed_time) {
            fprintf(stderr, "Reached time limit opt_allowed_time=%g\n", opt_allowed_time);
            terminate = true;
        }

        /* increase total maintenance time by the time spent on running this chunk of code */
        mainten_time += double(clock() - mainten_start) / CLOCKS_PER_SEC;
    }

    /* increase total number of calls for this logger */
    ++n_calls;

    /* return the state */
    return terminate;
}
