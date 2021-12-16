//
// Created by andreas on 05.10.21.
//

#ifndef SIMPLE_RAY_TRACING_RECTANGLE_H
#define SIMPLE_RAY_TRACING_RECTANGLE_H
#include "interfaces/i_rectangle.h"
#include "miscellaneous/validate.h"

class Rectangle final: IRectangle
{
public:
	Rectangle(Vector3D width_vector,
			  Vector3D height_vector,
			  const c_vector3 &bottom_left_position,
			  const Vector3D &normal);

	float width() const final;

	float height() const final;

	c_vector3 bottom_left_position() const final;

	~Rectangle() final = default;


	void set_material(std::shared_ptr<IMaterial> material) final;

	std::shared_ptr<IMaterial> get_material() const final;
	bool does_ray_intersect(std::shared_ptr<IRay> &ray, std::shared_ptr<IHitRecord> &hit_record) const final;

private:
	float width_;
	float height_;
	Vector3D width_vector_;
	Vector3D height_vector_;
	Point3D bottom_left_position_;
	Vector3D normal_;
	std::shared_ptr<IMaterial> material_;
	void init() const;
};


#endif //SIMPLE_RAY_TRACING_RECTANGLE_H
