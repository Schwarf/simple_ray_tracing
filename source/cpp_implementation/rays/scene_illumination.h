//
// Created by andreas on 16.10.21.
//

#ifndef SCENE_ILLUMINATION_H
#define SCENE_ILLUMINATION_H


#include "rays/interfaces/i_scene_illumination.h"
#include "miscellaneous/validate.h"
#include "objects/sphere.h"
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
	void add_light_source(const std::shared_ptr<ILightSource> &light_source) final;
	std::shared_ptr<ILightSource> light_source(size_t index) final;
	size_t number_of_light_sources() final;
	Color background_color(float parameter) final;
	void set_background_colors(const Color &color1, const Color &color2) final;
	void set_ground_sphere(const std::shared_ptr<ISphere> &ground_sphere) final;
	std::shared_ptr<ISphere> get_ground() override;
private:
	std::vector<std::shared_ptr<ILightSource>> light_source_vector_;
	std::shared_ptr<ISphere> ground_;
	Color background_color1_;
	Color background_color2_;

};


#endif //SCENE_ILLUMINATION_H
