//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_SPHERE_H
#define SIMPLE_RAY_TRACING_SPHERE_H
#include "objects/interfaces/i_sphere.h"
#include "check.h"
#include <algorithm>

class Sphere final : public ISphere{
public:
    Sphere(c_vector3 & center, float radius);
    Sphere(Sphere && ) =default;
    c_vector3 center() const final;

    float radius() const final;

    ~Sphere() override = default;

    bool does_ray_intersect(const IRay &ray, float &closest_hit_distance, c_vector3 & hit_point) const final;

    void set_material(std::shared_ptr<IMaterial> material) final;

    std::shared_ptr<IMaterial> get_material() final;

private:
    void init() const;
    c_vector3 center_;
    float radius_;
    std::shared_ptr<IMaterial> material_;
    Check check_;
};


#endif //SIMPLE_RAY_TRACING_SPHERE_H
