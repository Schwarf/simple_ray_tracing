//
// Created by andreas on 05.10.21.
//

#include "check.h"

void Check::is_above_threshold(const std::string &variable_name, const float &variable_value, const float &threshold,
                               const std::string &class_name) const {
    if (variable_value > threshold)
        return;
    const std::string message_part1 = "In class ";
    const std::string message_part2 = " is less than ";
    const std::string threshold_part = std::to_string(threshold);
    const std::string variable_part = std::to_string(variable_value);
    std::string message =
            message_part1 + class_name + " " + variable_name + "=" + variable_part + message_part2 + threshold_part;
    throw std::out_of_range(message);

}

void Check::is_below_threshold(const std::string &variable_name, const float &variable_value, const float &threshold,
                               const std::string &class_name) const {
    if (variable_value < threshold)
        return;
    const std::string message_part1 = "In class ";
    const std::string message_part2 = " is bigger than ";
    const std::string threshold_part = std::to_string(threshold);
    const std::string variable_part = std::to_string(variable_value);
    std::string message =
            message_part1 + class_name + " " + variable_name + "=" + variable_part + message_part2 + threshold_part;
    throw std::out_of_range(message);

}
