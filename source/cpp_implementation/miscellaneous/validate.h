//
// Created by andreas on 05.10.21.
//

#ifndef SIMPLE_RAY_TRACING_CHECK_H
#define SIMPLE_RAY_TRACING_CHECK_H

#include <stdexcept>


template <typename T>
class Validate
{
public:
	static void is_above_threshold(const std::string &variable_name,
							const T &variable_value,
							const T &threshold,
							const std::string &class_name)
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
	static void is_below_threshold(const std::string &variable_name,
							const T &variable_value,
							const T &threshold,
							const std::string &class_name)
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
	static void is_zero(const std::string &variable_name,
				 const T &variable_value,
				 const T &epsilon,
				 const std::string &class_name)
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
	static void is_not_zero(const std::string &variable_name,
					 const T &variable_value,
					 const T &epsilon,
					 const std::string &class_name)
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
	static void is_in_open_interval(const std::string &variable_name,
								  const T &variable_value,
								  const T &lower_bound,
								  const T &upper_bound,
							 	  const std::string &class_name)
	{
		const std::string interval_variable_name = " check-open-interval of " + variable_name;
		is_above_threshold(interval_variable_name, variable_value, lower_bound, class_name);
		is_below_threshold(interval_variable_name, variable_value, upper_bound, class_name);
	}
};


#endif //SIMPLE_RAY_TRACING_CHECK_H
