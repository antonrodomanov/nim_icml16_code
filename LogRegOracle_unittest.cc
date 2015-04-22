#include "LogRegOracle.h"
#include "gtest/gtest.h"
#include <vector>

TEST(SingleValTest, Basic) {
    std::vector<std::vector<double>> Z {
        {0.1, -0.01},
        {-0.01, -0.1},
        {0.3, 0.1}
    };
    std::vector<double> w {0.1, -0.4};
    double lambda = 0.1;

    LogRegOracle func(Z, lambda);

    EXPECT_NEAR(func.single_val(w, 1), 0.721337293512, 1e-5);
}

TEST(SingleGradTest, Basic) {
    std::vector<std::vector<double>> Z {
        {0.1, -0.01},
        {-0.01, -0.1},
        {0.3, 0.1}
    };
    std::vector<double> w {0.1, -0.4};
    double lambda = 0.1;

    LogRegOracle func(Z, lambda);

    std::vector<double> gi = func.single_grad(w, 1);
    EXPECT_NEAR(gi[0], 0.00490251, 1e-5);
    EXPECT_NEAR(gi[1], -0.09097488, 1e-5);
}
