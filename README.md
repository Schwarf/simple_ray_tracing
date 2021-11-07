# simple_ray_tracing
- [ray intersection](https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection)
- [phong model](https://en.wikipedia.org/wiki/Phong_reflection_model)
- [reflection and refraction](https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel)
- [Snells law](https://en.wikipedia.org/wiki/Snell%27s_law)

Build in "source" folder with cmake version >= 3.20:

cmake -Bcmake-build-debug -DCMAKE_BUILD_TYPE=DEBUG 

cmake --build cmake-build-debug/ -- -j 19

### ToDo
- The unit tests for cubic equation lack accuracy (even in double precision). 
  This might be a bug. The behaviour is reproducible for epsilon > 1.e-4 (quite large).
