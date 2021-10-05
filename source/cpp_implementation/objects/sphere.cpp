//
// Created by andreas on 03.10.21.
//

#include "sphere.h"

Sphere::Sphere(c_vector3 &center, float radius) :
        material_(nullptr) {
    center_ = center;
    radius_ = radius;
    init();
}

c_vector3 Sphere::center() const {
    return center_;
}

float Sphere::radius() const {
    return radius_;
}

void Sphere::init() const {
    check_.is_above_threshold("radius", radius_, 0.0, " Sphere");
}


bool Sphere::does_ray_intersect(const IRay &ray, float &closest_hit_distance, c_vector3 & hit_point) const {
    closest_hit_distance = -1.0;
    c_vector3 origin_to_center = (center_ - ray.origin());
    float origin_to_center_dot_direction = origin_to_center * ray.direction_normalized();
    if (origin_to_center_dot_direction < 0) {
        // Sphere center is behind ray origin
        return false;
    }

    float epsilon = 1e-5;
    float delta = origin_to_center_dot_direction * origin_to_center_dot_direction -
                  ((origin_to_center * origin_to_center) - radius_ * radius_);
    if (delta < 0.0) {
        return false;
    }
    if (std::abs(delta) < epsilon) {
        hit_point = ray.origin() + ray.direction_normalized() * closest_hit_distance;
        closest_hit_distance = origin_to_center_dot_direction;
        return true;
    }
    auto solution1 = origin_to_center_dot_direction - std::sqrt(delta);
    auto solution2 = origin_to_center_dot_direction + std::sqrt(delta);
    closest_hit_distance = solution1;

    if (closest_hit_distance < 0.0) {
        closest_hit_distance = solution2;
    }
    if (closest_hit_distance < 0.0) {
        return false;
    }
    hit_point = ray.origin() + ray.direction_normalized() * closest_hit_distance;
    return true;
}

void Sphere::set_material(std::shared_ptr<IMaterial> material) {
    material_ = material;
}

std::shared_ptr<IMaterial> Sphere::get_material() {
    return material_;
}

