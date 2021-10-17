//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_RAY_H
#define SIMPLE_RAY_TRACING_I_RAY_H
#include "miscellaneous/templates/c_vector.h"
class IRay
{
public:
	virtual c_vector3 direction_normalized() const = 0;
	virtual c_vector3 origin() const = 0;
	virtual ~IRay() = default;
};
#endif //SIMPLE_RAY_TRACING_I_RAY_H
