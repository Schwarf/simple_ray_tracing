//
// Created by andreas on 16.10.21.
//

#ifndef SCENE_ILLUMINATION_H
#define SCENE_ILLUMINATION_H


#include "miscellaneous/validate.h"
#include "light_source.h"
#include "objects/sphere.h"
#include <algorithm>
#include <vector>

class SceneIllumination
{
public:
	SceneIllumination() = default;
	explicit SceneIllumination(const LightSource &light_source);
	SceneIllumination(const SceneIllumination &rhs);
	SceneIllumination(SceneIllumination &&rhs) noexcept;
	SceneIllumination &operator=(const SceneIllumination &rhs);
	SceneIllumination &operator=(SceneIllumination &&rhs) noexcept;
	~SceneIllumination() = default;
	void add_light_source(const LightSource &light_source);
	void light_source(size_t index, LightSource & lightSource) const;
	size_t number_of_light_sources() const;
	Color background_color(float parameter) const;
	void set_background_colors(const Color &color1, const Color &color2);
	void set_ground_sphere(const ISpherePtr &ground_sphere);
	ISpherePtr get_ground() const;
private:
	std::vector<LightSource> light_source_vector_;
	ISpherePtr ground_;
	Color background_color1_;
	Color background_color2_;

};


#endif //SCENE_ILLUMINATION_H
