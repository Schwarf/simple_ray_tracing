
add_library(c_vector c_vector.h material.cpp material.h material_builder.cpp material_builder.h )

add_library(interfaces INTERFACE interfaces/i_reflection_coefficients.h interfaces/i_material_builder.h ./interfaces/i_check.h interfaces/i_specular_exponent.h interfaces/i_rgb_color.h interfaces/i_material.h interfaces/i_refraction_coefficient.h)