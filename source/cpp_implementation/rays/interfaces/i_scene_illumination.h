//
// Created by andreas on 16.10.21.
//

#ifndef I_SCENE_ILLUMINATION_H
#define I_SCENE_ILLUMINATION_H
#include "rays/interfaces/i_light_source.h"
#include <memory>
class ISceneIllumination{
public:
	virtual void add_light_source(std::shared_ptr<ILightSource> light_source) = 0;
	virtual std::shared_ptr<ILightSource> light_source(size_t index) = 0;
	virtual size_t number_of_light_sources() =0;
	virtual c_vector3 background_color() = 0;
	virtual void set_background_color(const c_vector3 & color) = 0;
	virtual ~ISceneIllumination() =default;
};

#endif //I_SCENE_ILLUMINATION_H
