//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_SPHERE_H
#define SIMPLE_RAY_TRACING_SPHERE_H
#include "interfaces/i_sphere.h"
#include "interfaces/i_check.h"
#include <algorithm>
class Sphere : public ISphere, public ICheck{
public:
    Sphere(c_vector3 & center, float radius);
    c_vector3 center() const final;

    float radius() const final;

    ~Sphere() override = default;

    void is_above_threshold(const std::string &variable_name, const float &variable_value,
                            const float &threshold) const final;

    bool does_ray_intersect(const IRay &ray, float &closest_hit_distance) const final;

private:
    void init() const;
    c_vector3 center_;
    float radius_;
};


#endif //SIMPLE_RAY_TRACING_SPHERE_H
