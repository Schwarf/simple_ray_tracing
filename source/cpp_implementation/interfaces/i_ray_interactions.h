//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_RAY_REFLECTION_H
#define SIMPLE_RAY_TRACING_I_RAY_REFLECTION_H
#include "c_vector.h"
class IRayInteractions{
public:
    virtual c_vector3 reflection(c_vector3 & light_direction, c_vector3 & point_normal) const = 0;
};

#endif //SIMPLE_RAY_TRACING_I_RAY_REFLECTION_H
