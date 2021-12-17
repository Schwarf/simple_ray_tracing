//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_RAY_H
#define SIMPLE_RAY_TRACING_I_RAY_H
#include "miscellaneous/templates/n_tuple.h"

class IRay
{
public:
	virtual Vector3D direction_normalized() const = 0;
	virtual Point3D origin() const = 0;
	virtual ~IRay() = default;
};
#endif //SIMPLE_RAY_TRACING_I_RAY_H
