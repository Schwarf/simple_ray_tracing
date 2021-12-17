//
// Created by andreas on 05.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_RECTANGLE_H
#define SIMPLE_RAY_TRACING_I_RECTANGLE_H

#include "miscellaneous/templates/n_tuple.h"
#include "i_target_object.h"

class IRectangle: public ITargetObject
{
public:
	virtual float width() const = 0;

	virtual float height() const = 0;

	virtual Point3D bottom_left_position() const = 0;

	virtual ~IRectangle() = default;
};

#endif //SIMPLE_RAY_TRACING_I_RECTANGLE_H

