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
			  const Point3D &bottom_left_position);

	float width() const final;

	float height() const final;

	Point3D bottom_left_position() const final;

	~Rectangle() final = default;


	void set_material(const IMaterialPtr &material) final;

	IMaterialPtr get_material() const final;
	bool does_ray_intersect(Ray &ray, HitRecord &hit_record) const ;

private:
	float width_;
	float height_;
	Vector3D width_vector_;
	Vector3D height_vector_;
	Point3D bottom_left_position_;
	Vector3D normal_;
	IMaterialPtr material_;
	void init() const;
};


#endif //SIMPLE_RAY_TRACING_RECTANGLE_H
