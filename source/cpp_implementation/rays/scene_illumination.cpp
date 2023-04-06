//
// Created by andreas on 16.10.21.
//

#include "scene_illumination.h"

SceneIllumination::SceneIllumination(ILightSourcePtr light_source)
	: background_color1_(Color{1, 1, 1})
{
	light_source_vector_.push_back(light_source);
}

SceneIllumination::SceneIllumination(const SceneIllumination &rhs)
{
	std::cout << "SceneIllumination COPY" << std::endl;
	this->background_color1_ = rhs.background_color1_;
	this->background_color2_ = rhs.background_color2_;
	this->light_source_vector_ = rhs.light_source_vector_;
}

SceneIllumination::SceneIllumination(SceneIllumination &&rhs) noexcept
{
	std::cout << "SceneIllumination MOVE" << std::endl;
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
	std::cout << "SceneIllumination MOVE-ASSIGN" << std::endl;
	std::swap(light_source_vector_, rhs.light_source_vector_);
	return *this;
}

void SceneIllumination::add_light_source(const ILightSourcePtr &light_source)
{
	light_source_vector_.push_back(light_source);
}

ILightSourcePtr SceneIllumination::light_source(size_t index) const
{
	if (index >= light_source_vector_.size()) {
		return nullptr;
	}
	return light_source_vector_[index];
}
size_t SceneIllumination::number_of_light_sources() const
{
	return light_source_vector_.size();
}
Color SceneIllumination::background_color(float parameter) const
{

	return (1.f - parameter) * background_color1_ + parameter * background_color2_;
}
void SceneIllumination::set_background_colors(const Color &color1, const Color &color2)
{
	background_color1_ = color1;
	background_color2_ = color2;
}
void SceneIllumination::set_ground_sphere(const ISpherePtr &ground_sphere)
{
	ground_ = ground_sphere;
}

ISpherePtr SceneIllumination::get_ground() const
{
	return ground_;
}

