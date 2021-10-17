//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_SPHERE_H
#define SIMPLE_RAY_TRACING_I_SPHERE_H
#include "miscellaneous/templates/c_vector.h"
#include "rays/interfaces/i_ray_intersection.h"
#include "i_target_object.h"

class ISphere:  public ITargetObject{
public:
    virtual c_vector3 center() const =0;
    virtual float radius() const =0;
    virtual ~ISphere() =default;
};
#endif //SIMPLE_RAY_TRACING_I_SPHERE_H
