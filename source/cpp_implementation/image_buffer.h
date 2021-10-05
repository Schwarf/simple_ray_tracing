//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_IMAGE_BUFFER_H
#define SIMPLE_RAY_TRACING_IMAGE_BUFFER_H

#include <memory>
#include "interfaces/i_image_buffer.h"
#include "c_vector.h"
class ImageBuffer final: public IImageBuffer {

public:
    ImageBuffer(int width, int height);
    int width() const final;

    int height() const final;

    std::unique_ptr<std::vector<c_vector3>> buffer() final;

    void set_pixel_value(size_t width_index, size_t height_index, c_vector3 pixel_color_value) final;

    c_vector3 get_pixel(size_t index) final;

private:
    std::vector<c_vector3> buffer_;
    int width_;
    int height_;
};


#endif //SIMPLE_RAY_TRACING_IMAGE_BUFFER_H
