//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_IMAGE_BUFFER_H
#define SIMPLE_RAY_TRACING_I_IMAGE_BUFFER_H
#include "c_vector.h"
#include <vector>

class IImageBuffer{
public:
    virtual int width() const = 0;
    virtual int height() const= 0;
    virtual void set_pixel_value(size_t width_index, size_t height_index, const c_vector3 pixel_color_value) = 0;
    virtual c_vector3 get_rgb_pixel(size_t index) = 0;
    virtual c_vector3 get_rgb_pixel(int width_index, int height_index) = 0;
    virtual std::unique_ptr<std::vector<c_vector3>> buffer() = 0;

};
#endif //SIMPLE_RAY_TRACING_I_IMAGE_BUFFER_H
