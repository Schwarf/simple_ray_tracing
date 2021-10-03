//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_RAY_H
#define SIMPLE_RAY_TRACING_RAY_H

#include "interfaces/i_ray.h"

class Ray : public IRay{
public:
    Ray(c_vector3 & origin, c_vector3 & direction);
    c_vector3 direction_normalized() const override;

    c_vector3 origin() const override;

    ~Ray() override = default;
private:
    c_vector3 direction_normalized_;
    c_vector3 origin_;
};


#endif //SIMPLE_RAY_TRACING_RAY_H
