//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_IMAGE_BUFFER_H
#define SIMPLE_RAY_TRACING_IMAGE_BUFFER_H

#include <memory>
#include "interfaces/i_image_buffer.h"
#include "c_vector.h"
class ImageBuffer: public IImageBuffer {

public:
    ImageBuffer(int width, int height);
    int width() const override;

    int height() const override;

    std::vector<c_vector3> buffer() override;
private:
    std::vector<c_vector3> buffer_;
    int width_;
    int height_;
};


#endif //SIMPLE_RAY_TRACING_IMAGE_BUFFER_H
