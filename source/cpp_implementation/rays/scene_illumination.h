//
// Created by andreas on 16.10.21.
//

#ifndef SCENE_ILLUMINATION_H
#define SCENE_ILLUMINATION_H


#include <rays/interfaces/i_scene_illumination.h>
#include <algorithm>
#include <vector>

class SceneIllumination: public ISceneIllumination
{
public:
	SceneIllumination() = default;
	explicit SceneIllumination(std::shared_ptr<ILightSource> light_source);
	SceneIllumination(const SceneIllumination &rhs);
	SceneIllumination(SceneIllumination &&rhs) noexcept;
	SceneIllumination &operator=(const SceneIllumination &rhs);
	SceneIllumination &operator=(SceneIllumination &&rhs) noexcept;
	~SceneIllumination() override = default;
	void add_light_source(std::shared_ptr<ILightSource> light_source) final;
	std::shared_ptr<ILightSource> light_source(size_t index) final;
	size_t number_of_light_sources() final;
	c_vector3 background_color() final;
	void set_background_color(const c_vector3 &color) final;

private:
	std::vector<std::shared_ptr<ILightSource>> light_source_vector_;
	c_vector3 background_color_;

};


#endif //SCENE_ILLUMINATION_H
