//
// Created by andreas on 05.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_RECTANGLE_H
#define SIMPLE_RAY_TRACING_I_RECTANGLE_H

#include "miscellaneous/templates/c_vector.h"
#include "i_target_object.h"

class IRectangle : public ITargetObject {
public:
    virtual float width() const = 0;

    virtual float height() const = 0;

    virtual c_vector3 bottom_left_position() const = 0;

    virtual ~IRectangle() = default;
};

#endif //SIMPLE_RAY_TRACING_I_RECTANGLE_H

