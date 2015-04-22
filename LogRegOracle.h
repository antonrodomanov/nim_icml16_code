#ifndef LOG_REG_ORACLE_H_
#define LOG_REG_ORACLE_H_

#include <vector>

class LogRegOracle
{
public:
    LogRegOracle(const std::vector<std::vector<double>>& Z, double lambda);

    std::vector<double> single_grad(std::vector<double> w, int idx);

private:
    const std::vector<std::vector<double>>& Z;
    double lambda;
};

#endif
