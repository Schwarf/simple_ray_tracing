//
// Created by andreas on 04.10.21.
//

#include "ray_interactions.h"

std::shared_ptr<IRay> RayInteractions::reflected_ray(const std::shared_ptr<IRay> &ray, const std::shared_ptr<IHitRecord> &hit_record) const
{
	Vector3D reflected_ray_direction = ray->direction_normalized() - hit_record->hit_normal() * 2.f * (
		ray->direction_normalized()*hit_record->hit_normal()) + UniformRandomNumberGenerator::random_vector_in_unit_sphere<float>()
		    * UniformRandomNumberGenerator::get_random<float>(0.f, 0.1f);
	return std::make_shared<Ray>(Ray(hit_record->hit_point(), reflected_ray_direction));

}
std::shared_ptr<IRay>  RayInteractions::refracted_ray(const std::shared_ptr<IRay> &ray,
													  const std::shared_ptr<IHitRecord> &hit_record,
													  const float &air_refraction_index) const
{
	float cosine = -std::max(-1.f, std::min(1.f, ray->direction_normalized()*hit_record->hit_normal()));
	auto hit_normal = hit_record->hit_normal();
	auto material_refraction_index = hit_record->material()->refraction_index();
	auto air_index = air_refraction_index;
	if(cosine < 0) {
		// ray is inside sphere, switch refraction_indices and normal
		hit_normal = -1.f*hit_normal;
		auto help = material_refraction_index;
		material_refraction_index = air_refraction_index;
		air_index = help;
	}
	float ratio = air_index/material_refraction_index;
	float k = 1.f - ratio*ratio*(1.f - cosine*cosine);
	auto refracted_ray_direction = UniformRandomNumberGenerator::random_vector_in_unit_sphere<float>() *
	    					 UniformRandomNumberGenerator::get_random<float>(0.f, 0.1f);
	if (k > 0) {
		refracted_ray_direction = ray->direction_normalized()*ratio + hit_normal*(ratio*cosine - std::sqrt(k));
	}
	return std::make_shared<Ray>(Ray(hit_record->hit_point(), refracted_ray_direction));
}
