#ifndef AUXILIARY_H
#define AUXILIARY_H

#include <vector>
#include <string>

void read_svmlight_file(const std::string& path, int N, int D, std::vector<std::vector<double>>& X, std::vector<int>& y);

std::vector<std::vector<double>> transform_to_z(std::vector<std::vector<double>> X, std::vector<int> y);

#endif
