#ifndef LOG_REG_ORACLE_H_
#define LOG_REG_ORACLE_H_

#include <vector>

class LogRegOracle
{
public:
    LogRegOracle(const std::vector<std::vector<double>>& Z, double lambda);

    double single_val(const std::vector<double>& w, int idx);
    std::vector<double> single_grad(const std::vector<double>& w, int idx);

    double full_val(const std::vector<double>& w);

private:
    const std::vector<std::vector<double>>& Z;
    double lambda;
};

#endif
