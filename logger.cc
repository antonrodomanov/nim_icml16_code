#include "logger.h"
#include <vector>
#include <cstdio>
#include "special.h"
#include <ctime>

Logger::Logger(const LogRegOracle& func) :
    func(func),
    how_often(func.n_samples() / 10),
    n_calls(0),
    t_start(clock()),
    mainten_time(0.0)
    {}

void Logger::log(const std::vector<double>& w)
{
    if (n_calls % how_often == 0) {
        /* for counting time spent on this maintenance code */
        clock_t mainten_start = clock();

        double epoch = double(n_calls) / func.n_samples();
        double f = func.full_val(w);
        double norm_g = infnorm(func.full_grad(w));
        double elaps = double(clock() - t_start) / CLOCKS_PER_SEC;
        /* this code is just for maintenance, don't add its time to the optimiser's time */
        elaps -= mainten_time;

        /* always start elaps at zero, first call is for free */
        if (n_calls == 0) {
            elaps = 0;
        }

        fprintf(stderr, "%9.2f %9.2f %15.6e %15.6e\n", epoch, elaps, f, norm_g);

        trace_epoch.push_back(epoch);
        trace_elaps.push_back(elaps);
        trace_val.push_back(f);
        trace_norm_grad.push_back(norm_g);

        mainten_time += double(clock() - mainten_start) / CLOCKS_PER_SEC;
    }
    ++n_calls;
}
