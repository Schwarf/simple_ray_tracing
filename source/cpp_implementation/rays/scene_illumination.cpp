//
// Created by andreas on 16.10.21.
//

#include "scene_illumination.h"

SceneIllumination::SceneIllumination(std::shared_ptr<ILightSource> light_source)
{
	background_color1_ = c_vector3{1, 1, 1};
	background_color1_ = c_vector3{1, 1, 1};
	light_source_vector_.push_back(light_source);
}

SceneIllumination::SceneIllumination(const SceneIllumination &rhs)
{
	this->background_color1_ = rhs.background_color1_;
	this->background_color2_ = rhs.background_color2_;
	this->light_source_vector_ = rhs.light_source_vector_;
}

SceneIllumination::SceneIllumination(SceneIllumination &&rhs) noexcept
{
	this->light_source_vector_ = std::move(rhs.light_source_vector_);
	this->background_color1_ = rhs.background_color1_;
	this->background_color2_ = rhs.background_color2_;
}
SceneIllumination &SceneIllumination::operator=(const SceneIllumination &rhs)
{
	return *this = SceneIllumination(rhs);
}
SceneIllumination &SceneIllumination::operator=(SceneIllumination &&rhs) noexcept
{
	std::swap(light_source_vector_, rhs.light_source_vector_);
	return *this;
}

void SceneIllumination::add_light_source(std::shared_ptr<ILightSource> light_source)
{
	light_source_vector_.push_back(light_source);
}

std::shared_ptr<ILightSource> SceneIllumination::light_source(size_t index)
{
	if (index >= light_source_vector_.size()) {
		return nullptr;
	}
	return light_source_vector_[index];
}
size_t SceneIllumination::number_of_light_sources()
{
	return light_source_vector_.size();
}
c_vector3 SceneIllumination::background_color(const float parameter)
{

	return (1.f-parameter)*background_color1_ + parameter*background_color2_;
}
void SceneIllumination::set_background_colors(const c_vector3 &color1, const c_vector3 &color2)
{
	background_color1_ = color1;
	background_color2_ = color2;
}
void SceneIllumination::set_ground_sphere(const std::shared_ptr<ISphere> &ground_sphere)
{
	ground_ = ground_sphere;
}

std::shared_ptr<ISphere> SceneIllumination::get_ground()
{
	return ground_;
}

