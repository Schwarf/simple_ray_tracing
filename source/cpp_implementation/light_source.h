//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_LIGHT_SOURCE_H
#define SIMPLE_RAY_TRACING_LIGHT_SOURCE_H

#include "interfaces/i_light_source.h"

class LightSource : public ILightSource {
public:
    LightSource(const c_vector3 &position, float intensity);

    c_vector3 position() const final;

    float intensity() const final;

private:
    c_vector3 position_;
    float intensity_;
};


#endif //SIMPLE_RAY_TRACING_LIGHT_SOURCE_H
