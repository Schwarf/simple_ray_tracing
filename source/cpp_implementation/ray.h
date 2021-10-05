//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_RAY_H
#define SIMPLE_RAY_TRACING_RAY_H

#include "interfaces/i_ray.h"

class Ray final : public IRay{
public:
    Ray(c_vector3 & origin, c_vector3 & direction);
    c_vector3 direction_normalized() const final;

    c_vector3 origin() const final;

    ~Ray() override = default;
private:
    c_vector3 direction_normalized_;
    c_vector3 origin_;
};


#endif //SIMPLE_RAY_TRACING_RAY_H
