//
// Created by andreas on 04.10.21.
//

#include "ray_interactions.h"


c_vector3 RayInteractions::reflection(const c_vector3 &light_direction, const c_vector3 &point_normal) const
{
	return light_direction - point_normal * 2.f * (light_direction * point_normal);
}
c_vector3 RayInteractions::refraction(const c_vector3 &light_direction,
									  const c_vector3 &point_normal,
									  const float &material_refraction_index, const float &air_refraction_index) const
{
	float cosine = -std::max(-1.f, std::min(1.f, light_direction*point_normal));
	if(cosine < 0) {
		return this->refraction(light_direction, -point_normal, air_refraction_index, material_refraction_index);
	}
	float ratio = air_refraction_index/material_refraction_index;
	float k = 1.f - ratio*ratio*(1.f - cosine*cosine);
	auto result = c_vector3{1.,0.,0.,};
	if (k > 0) {
		result = light_direction*ratio + point_normal*(ratio*cosine - std::sqrt(k));
	}
	return result;

}
