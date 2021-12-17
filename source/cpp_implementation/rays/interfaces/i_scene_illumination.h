//
// Created by andreas on 16.10.21.
//

#ifndef I_SCENE_ILLUMINATION_H
#define I_SCENE_ILLUMINATION_H
#include "rays/interfaces/i_light_source.h"
#include "materials/interfaces/i_material.h"
#include <memory>
#include <objects/interfaces/i_sphere.h>
class ISceneIllumination
{
public:
	virtual void add_light_source(const std::shared_ptr<ILightSource> &light_source) = 0;
	virtual std::shared_ptr<ILightSource> light_source(size_t index) = 0;
	virtual size_t number_of_light_sources() = 0;
	virtual Color background_color(float parameter) = 0;
	virtual void set_background_colors(const Color &color1, const Color &color2) = 0;
	virtual void set_ground_sphere(const std::shared_ptr<ISphere> &ground_sphere) = 0;
	virtual std::shared_ptr<ISphere> get_ground() = 0;
	virtual ~ISceneIllumination() = default;
};

#endif //I_SCENE_ILLUMINATION_H
