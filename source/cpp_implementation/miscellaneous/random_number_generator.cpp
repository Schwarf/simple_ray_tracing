//
// Created by andreas on 16.12.21.
//
#include "random_number_generator.h"

std::mt19937 UniformRandomNumberGenerator::generator_ = std::mt19937 (time(0));