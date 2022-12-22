//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_RAY_H
#define SIMPLE_RAY_TRACING_I_RAY_H
#include <memory>
#include "miscellaneous/templates/fixed_sized_array.h"

class IRay
{
public:
	virtual Vector3D direction_normalized() const = 0;
	virtual Point3D origin() const = 0;
	virtual void set_direction(const Vector3D&  direction) = 0;
	virtual void set_origin(const Point3D & origin) = 0;
	virtual ~IRay() = default;
};

using IRayPtr = std::shared_ptr<IRay>;
#endif //SIMPLE_RAY_TRACING_I_RAY_H
