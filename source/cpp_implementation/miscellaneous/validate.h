//
// Created by andreas on 05.10.21.
//

#ifndef SIMPLE_RAY_TRACING_CHECK_H
#define SIMPLE_RAY_TRACING_CHECK_H

#include "miscellaneous/interfaces/i_validate.h"
#include <stdexcept>


template <typename T>
class Validate final: public IValidate<T>
{
public:
	~Validate() override = default;
	void is_above_threshold(const std::string &variable_name,
							const T &variable_value,
							const T &threshold,
							const std::string &class_name) const override
	{
		if (variable_value > threshold)
			return;
		const std::string message_part1 = "In class ";
		const std::string message_part2 = " is lower than ";
		const std::string threshold_part = std::to_string(threshold);
		const std::string variable_part = std::to_string(variable_value);
		std::string message =
			message_part1 + class_name + " " + variable_name + "=" + variable_part + message_part2 + threshold_part;
		throw std::out_of_range(message);

	}
	void is_below_threshold(const std::string &variable_name,
							const T &variable_value,
							const T &threshold,
							const std::string &class_name) const override
	{
		if (variable_value < threshold)
			return;
		const std::string message_part1 = "In class ";
		const std::string message_part2 = " is greater than ";
		const std::string threshold_part = std::to_string(threshold);
		const std::string variable_part = std::to_string(variable_value);
		std::string message =
			message_part1 + class_name + " " + variable_name + "=" + variable_part + message_part2 + threshold_part;
		throw std::out_of_range(message);

	}
	void is_zero(const std::string &variable_name,
				 const T &variable_value,
				 const T &epsilon,
				 const std::string &class_name) const override
	{
		if (variable_value > -epsilon && variable_value < epsilon )
			return;
		const std::string message_part1 = "In class ";
		const std::string message_part2 = " is not zero (within epsilon) ";
		const std::string threshold_part = std::to_string(epsilon);
		const std::string variable_part = std::to_string(variable_value);
		std::string message =
			message_part1 + class_name + " " + variable_name + "=" + variable_part + message_part2 + threshold_part;
		throw std::out_of_range(message);

	}
	void is_not_zero(const std::string &variable_name,
					 const T &variable_value,
					 const T &epsilon,
					 const std::string &class_name) const override
	{
		if (variable_value < -epsilon || variable_value > epsilon )
			return;
		const std::string message_part1 = "In class ";
		const std::string message_part2 = " is zero (within epsilon) ";
		const std::string threshold_part = std::to_string(epsilon);
		const std::string variable_part = std::to_string(variable_value);
		std::string message =
			message_part1 + class_name + " " + variable_name + "=" + variable_part + message_part2 + threshold_part;
		throw std::out_of_range(message);
	}
};


#endif //SIMPLE_RAY_TRACING_CHECK_H
