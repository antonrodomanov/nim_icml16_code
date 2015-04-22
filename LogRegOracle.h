#ifndef LOG_REG_ORACLE_H_
#define LOG_REG_ORACLE_H_

#include <vector>

class LogRegOracle
{
public:
    LogRegOracle(const std::vector<std::vector<double>>& Z, double lambda);

    int n_samples() const;

    double single_val(const std::vector<double>& w, int idx) const;
    std::vector<double> single_grad(const std::vector<double>& w, int idx) const;

    double full_val(const std::vector<double>& w) const;
    std::vector<double> full_grad(const std::vector<double>& w) const;

private:
    const std::vector<std::vector<double>>& Z;
    double lambda;
};

#endif
