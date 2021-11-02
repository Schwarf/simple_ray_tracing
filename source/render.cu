//
// Created by andreas on 01.11.21.
//

#include "cuda_implementation/miscellaneous/templates/c_vector.h"
#include <fstream>

#define checkCudaErrors(value) check_cuda( (value), #value, __FILE__, __LINE__)

__global__ void render_it(c_vector3 *buffer, size_t max_width, size_t max_height)
{
	//size_t width = threadIdx.x + blockIdx.x * blockDim.x;
	//size_t height = threadIdx.y + blockIdx.y * blockDim.y;
	size_t width = threadIdx.x;
	size_t height = blockIdx.x;
	if ((width >= max_width) || (height >= max_height)) {
		return;
	}
	size_t pixel_index = height * max_width + width;
	buffer[pixel_index] = c_vector3{0.2, 0.7, 0.8};
}

int main()
{
	size_t width = 1024;
	size_t height = 768;
	// Why is 32 the maximum number of threads per block
	constexpr size_t threads_per_block = 32;
	//dim3 number_of_threads(threads_per_block, threads_per_block);

	//dim3 number_of_blocks(width / threads_per_block, height / threads_per_block);
	int number_of_blocks = 768;
	int number_of_threads{1024};
	size_t buffer_size = width * height * sizeof(float3);
	std::cout << buffer_size << std::endl;
	c_vector3 *buffer;
	cudaMallocManaged((void **)&buffer, buffer_size);

	render_it<<<number_of_blocks, number_of_threads>>>(buffer, width, height);
	cudaGetLastError();
	cudaDeviceSynchronize();
	std::ofstream ofs;
	ofs.open("./cuda_image.ppm");
	ofs << "P6\n" << width << " " << height << "\n255\n";
	for (size_t pixel_index = 0; pixel_index < width * height; ++pixel_index) {
		for (size_t color_index = 0; color_index < 3; color_index++) {
			ofs << static_cast<char>(255 * std::max(0.f, std::min(1.f, buffer[pixel_index][color_index])));
		}
	}

	return 0;
}