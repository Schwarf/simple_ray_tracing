//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_LIGHT_SOURCE_H
#define SIMPLE_RAY_TRACING_I_LIGHT_SOURCE_H
#include "c_vector.h"

class ILightSource{
public:
    virtual c_vector3 position() const = 0;
    virtual float intensity() const = 0;
};

#endif //SIMPLE_RAY_TRACING_I_LIGHT_SOURCE_H
