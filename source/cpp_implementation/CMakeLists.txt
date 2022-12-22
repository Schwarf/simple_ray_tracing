add_library(ray_lib STATIC rays/ray_interactions.cpp rays/ray_interactions.h rays/ray.cpp rays/ray.h
        rays/scene_illumination.cpp rays/scene_illumination.h rays/light_source.cpp rays/light_source.h rays/hit_record.cpp rays/hit_record.h)

add_library(ray_interfaces INTERFACE rays/interfaces/i_ray_intersection.h rays/interfaces/i_ray.h rays/interfaces/i_ray_interactions.h
        rays/interfaces/i_scene_illumination.h rays/interfaces/i_light_source.h)

add_library(misc_lib STATIC miscellaneous/validate.h miscellaneous/templates/fixed_sized_array.h
            miscellaneous/image_buffer.cpp miscellaneous/image_buffer.h miscellaneous/quartic_equation.h miscellaneous/random_number_generator.h miscellaneous/random_number_generator.cpp)
add_library(misc_interfaces INTERFACE miscellaneous/interfaces/i_image_buffer.h miscellaneous/quadratic_equation.h)

add_library(materials_lib STATIC materials/material.cpp materials/material.h materials/material_builder.cpp
        materials/material_builder.h )

add_library(objects_lib STATIC objects/sphere.cpp objects/sphere.h objects/rectangle.cpp objects/rectangle.h objects/object_list.cpp objects/object_list.h create_scenes/create_object_list.cpp create_scenes/create_object_list.h)
add_library(objects_interfaces INTERFACE objects/interfaces/i_object_list.h objects/interfaces/i_sphere.h
            objects/interfaces/i_target_object.h objects/interfaces/i_rectangle.h)

add_library(materials_interfaces INTERFACE materials/interfaces/i_phong_reflection_coefficients.h materials/interfaces/i_material_builder.h
        materials/interfaces/i_material.h
        materials/interfaces/i_refraction_coefficients.h)

add_library(scenes_lib STATIC create_scenes/create_object_list.cpp create_scenes/create_object_list.h create_scenes/create_scene_illumination.cpp create_scenes/create_scene_illumination.h)

add_library(camera_interfaces INTERFACE camera/interfaces/i_camera.h rays/interfaces/i_hit_record.h)
add_library(camera_lib STATIC camera/camera.cpp camera/camera.h)

target_link_libraries(scenes_lib objects_lib materials_lib ray_lib)
target_link_libraries(camera_lib scenes_lib ray_lib objects_lib misc_lib)