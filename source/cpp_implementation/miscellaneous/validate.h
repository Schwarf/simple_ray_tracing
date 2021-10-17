//
// Created by andreas on 05.10.21.
//

#ifndef SIMPLE_RAY_TRACING_CHECK_H
#define SIMPLE_RAY_TRACING_CHECK_H

#include "miscellaneous/interfaces/i_validate.h"
#include <stdexcept>


class Validate final: public IValidate
{
public:
	void is_above_threshold(const std::string &variable_name, const float &variable_value, const float &threshold,
							const std::string &class_name) const final;

	void is_below_threshold(const std::string &variable_name, const float &variable_value, const float &threshold,
							const std::string &class_name) const final;

	~Validate() override = default;
};


#endif //SIMPLE_RAY_TRACING_CHECK_H
