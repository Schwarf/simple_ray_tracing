//
// Created by andreas on 22.11.21.
//

#ifndef I_CAMERA_H
#define I_CAMERA_H
#include "objects/interfaces/i_object_list.h"

#include <memory>

class ICamera
{
public:
	virtual void render_image(const IObjectListPtr &objects_in_scene,
							  const SceneIllumination &scene_illumination) = 0;
	virtual Ray get_camera_ray(float width_coordinate, float height_coordinate) = 0;
	virtual float aspect_ratio() = 0;
	virtual float focal_length() = 0;
	virtual IImageBufferPtr get_image_buffer()=0;
	virtual void enable_antialiasing() = 0;
	virtual void disable_antialiasing() = 0;
	virtual bool antialiasing_enabled() = 0;

private:
	virtual Color get_pixel_color(Ray & camera_ray,
								  const IObjectListPtr &objects_in_scene,
								  const SceneIllumination &scene_illumination,
								  size_t recursion_depth) = 0;
};

#endif //I_CAMERA_H
