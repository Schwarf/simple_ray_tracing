add_library(ray STATIC rays/ray_interactions.cpp rays/ray_interactions.h rays/ray.cpp rays/ray.h)
add_library(ray_interfaces INTERFACE rays/interfaces/i_ray_intersection.h rays/interfaces/i_ray.h rays/interfaces/i_ray_interactions.h)

add_library(misc STATIC miscellaneous/check.cpp miscellaneous/check.h miscellaneous/templates/c_vector.h)
add_library(misc_interfaces INTERFACE miscellaneous/interfaces/i_check.h)

add_library(c_vector STATIC materials/material.cpp materials/material.h materials/material_builder.cpp
        materials/material_builder.h miscellaneous/image_buffer.cpp miscellaneous/image_buffer.h objects/sphere.cpp objects/sphere.h
        objects/interfaces/i_geometric_object.h rays/light_source.cpp rays/light_source.h
        objects/interfaces/i_rectangle.h objects/rectangle.cpp objects/rectangle.h)
add_library(interfaces INTERFACE materials/interfaces/i_reflection_coefficients.h materials/interfaces/i_material_builder.h
        materials/interfaces/i_specular_exponent.h materials/interfaces/i_material.h
        materials/interfaces/i_refraction_coefficient.h miscellaneous/interfaces/i_image_buffer.h objects/interfaces/i_sphere.h
        rays/interfaces/i_light_source.h)

