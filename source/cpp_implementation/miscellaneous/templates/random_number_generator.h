//
// Created by andreas on 22.11.21.
//

#ifndef RANDOM_NUMBER_GENERATOR_H
#define RANDOM_NUMBER_GENERATOR_H
#include <type_traits>
#include "c_vector.h"
class UniformRandomNumberGenerator
{
public:
	UniformRandomNumberGenerator(){
		generator_ =  std::mt19937(time(0));
	}

	template <typename T, typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
	inline T get_random(const T & lower_bound, const T &upper_bound)
	{
		auto real_distribution_ = std::uniform_real_distribution<T>(lower_bound, upper_bound);
		return real_distribution_(generator_);
	}

	template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
	inline T get_random(const T & lower_bound, const T & upper_bound)
	{
		auto int_distribution_ = std::uniform_int_distribution<T>(lower_bound, upper_bound);
		return int_distribution_(generator_);
	}

	template <typename T>
	inline c_vector3 random_vector_in_unit_sphere()
	{
		auto lower_bound = -1;
		auto upper_bound = 1;
		while(true) {
			c_vector3 random_vector = {get_random<T>(lower_bound, upper_bound),
									   get_random<T>(lower_bound, upper_bound),
									   get_random<T>(lower_bound, upper_bound)};
			if (random_vector.norm() < 1)
				return random_vector;
		}
	}


private:
	std::mt19937 generator_;

};


#endif //RANDOM_NUMBER_GENERATOR_H
