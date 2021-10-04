//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_GEOMETRIC_OBJECT_H
#define SIMPLE_RAY_TRACING_I_GEOMETRIC_OBJECT_H
#include <memory>
#include "i_material.h"

class IGeometricObject{
public:
    virtual void set_material(std::shared_ptr<IMaterial> material) = 0;
    virtual std::shared_ptr<IMaterial> get_material() = 0;
};

#endif //SIMPLE_RAY_TRACING_I_GEOMETRIC_OBJECT_H
