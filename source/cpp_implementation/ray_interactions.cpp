//
// Created by andreas on 04.10.21.
//

#include "ray_interactions.h"


c_vector3 RayInteractions::reflection(c_vector3 &light_direction, c_vector3 &point_normal) const {
    return light_direction - point_normal*2.f * (light_direction*point_normal);
}

c_vector3 RayInteractions::casting(const IRay &ray, IGeometricObject &sphere, const ILightSource &light_source) const {
    return c_vector3{};
}
