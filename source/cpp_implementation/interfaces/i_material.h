//
// Created by andreas on 02.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_MATERIAL_H
#define SIMPLE_RAY_TRACING_I_MATERIAL_H

class Material {
public:
    virtual specular_reflection() = 0;

    virtual diffuse_reflection() = 0;

    virtual ambient_reflection() = 0;

    virtual shininess() = 0;
};

#endif //SIMPLE_RAY_TRACING_I_MATERIAL_H
