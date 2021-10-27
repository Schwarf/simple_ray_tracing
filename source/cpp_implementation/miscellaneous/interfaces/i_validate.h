//
// Created by andreas on 02.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_CHECK_H
#define SIMPLE_RAY_TRACING_I_CHECK_H
#include <string>

template <typename T>
class IValidate
{
public:
	virtual void
	is_above_threshold(const std::string &variable_name, const T &variable_value, const T &threshold,
					   const std::string &class_name) const = 0;
	virtual void
	is_below_threshold(const std::string &variable_name, const T &variable_value, const T &threshold,
					   const std::string &class_name) const = 0;
	virtual void
	is_zero(const std::string &variable_name, const T &variable_value, const T &epsilon,
						  const std::string &class_name) const =0;

	virtual void
	is_not_zero(const std::string &variable_name, const T &variable_value, const T &epsilon,
			const std::string &class_name) const =0;

	virtual ~IValidate() = default;
};

#endif //SIMPLE_RAY_TRACING_I_CHECK_H
