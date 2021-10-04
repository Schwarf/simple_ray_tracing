//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_RAY_INTERACTIONS_H
#define SIMPLE_RAY_TRACING_RAY_INTERACTIONS_H
#include "interfaces/i_ray_interactions.h"

class RayInteractions : IRayInteractions{
public:
    c_vector3 reflection(c_vector3 &light_direction, c_vector3 &point_normal) const final;
};


#endif //SIMPLE_RAY_TRACING_RAY_INTERACTIONS_H
