//
// Created by andreas on 16.10.21.
//

#include "scene_illumination.h"

SceneIllumination::SceneIllumination(std::shared_ptr<ILightSource> light_source)
{
	background_color_ = c_vector3{0, 0, 0};
	light_source_vector_.push_back(light_source);
}

SceneIllumination::SceneIllumination(const SceneIllumination &rhs)
{
	this->background_color_ = rhs.background_color_;
	this->light_source_vector_ = rhs.light_source_vector_;
}

SceneIllumination::SceneIllumination(SceneIllumination &&rhs) noexcept
{
	this->light_source_vector_ = std::move(rhs.light_source_vector_);
	this->background_color_ = rhs.background_color_;
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
c_vector3 SceneIllumination::background_color()
{
	return background_color_;
}
void SceneIllumination::set_background_color(const c_vector3 &color)
{
	background_color_ = color;
}

