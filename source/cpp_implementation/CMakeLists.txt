


add_library(c_vector c_vector.h material.cpp material.h material_builder.cpp material_builder.h image_buffer.cpp
        image_buffer.h sphere.cpp sphere.h ray.cpp ray.h interfaces/i_geometric_object.h light_source.cpp light_source.h)
add_library(interfaces INTERFACE interfaces/i_reflection_coefficients.h interfaces/i_material_builder.h
        ./interfaces/i_check.h interfaces/i_specular_exponent.h interfaces/i_rgb_color.h interfaces/i_material.h
        interfaces/i_refraction_coefficient.h interfaces/i_image_buffer.h interfaces/i_sphere.h
        interfaces/i_light_source.h interfaces/i_ray_intersection.h interfaces/i_ray.h )

