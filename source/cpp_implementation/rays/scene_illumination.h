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
	explicit SceneIllumination(const ILightSourcePtr &light_source);
	SceneIllumination(const SceneIllumination &rhs);
	SceneIllumination(SceneIllumination &&rhs) noexcept;
	SceneIllumination &operator=(const SceneIllumination &rhs);
	SceneIllumination &operator=(SceneIllumination &&rhs) noexcept;
	~SceneIllumination() override = default;
	void add_light_source(const ILightSourcePtr &light_source) final;
	ILightSourcePtr light_source(size_t index) const final;
	size_t number_of_light_sources() const final;
	Color background_color(float parameter) const final;
	void set_background_colors(const Color &color1, const Color &color2) final;
	void set_ground_sphere(const ISpherePtr &ground_sphere) final;
	ISpherePtr get_ground() const final;
private:
	std::vector<ILightSourcePtr> light_source_vector_;
	ISpherePtr ground_;
	Color background_color1_;
	Color background_color2_;

};


#endif //SCENE_ILLUMINATION_H
