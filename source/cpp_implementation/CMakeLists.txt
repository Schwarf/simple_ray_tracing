


add_library(c_vector c_vector.h materials/material.cpp materials/material.h materials/material_builder.cpp materials/material_builder.h image_buffer.cpp
        image_buffer.h objects/sphere.cpp objects/sphere.h ray.cpp ray.h objects/interfaces/i_geometric_object.h light_source.cpp light_source.h)
add_library(interfaces INTERFACE materials/interfaces/i_reflection_coefficients.h materials/interfaces/i_material_builder.h
        ./interfaces/i_check.h materials/interfaces/i_specular_exponent.h materials/interfaces/i_rgb_color.h materials/interfaces/i_material.h
        materials/interfaces/i_refraction_coefficient.h interfaces/i_image_buffer.h objects/interfaces/i_sphere.h
        interfaces/i_light_source.h interfaces/i_ray_intersection.h interfaces/i_ray.h )

