//
// Created by andreas on 05.10.21.
//

#ifndef SIMPLE_RAY_TRACING_RECTANGLE_H
#define SIMPLE_RAY_TRACING_RECTANGLE_H
#include "interfaces/i_rectangle.h"
#include "check.h"

class Rectangle : IRectangle{
public:
    Rectangle(c_vector3 width_vector, c_vector3 height_vector, const c_vector3 & bottom_left_position, const c_vector3 & normal);

    float width() const final;

    float height() const final;

    c_vector3 bottom_left_position() const final;

    ~Rectangle() final = default;

    bool does_ray_intersect(const IRay &ray, float &closest_hit_distance, c_vector3 &hit_point) const final;

    void set_material(std::shared_ptr<IMaterial> material) final;

    std::shared_ptr<IMaterial> get_material() final;
private:
    float width_;
    float height_;
    c_vector3 width_vector_;
    c_vector3 height_vector_;
    c_vector3 bottom_left_position_;
    c_vector3 normal_;
    std::shared_ptr<IMaterial> material_;
    Check check_;
    void init() const;
};


#endif //SIMPLE_RAY_TRACING_RECTANGLE_H
