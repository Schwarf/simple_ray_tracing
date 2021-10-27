//
// Created by andreas on 11.10.21.
//

#ifndef CUBIC_EQUATION_H
#define CUBIC_EQUATION_H

#include "interfaces/i_solve.h"
#include "quadratic_equation.h"
#include "validate.h"
#include "stdexcept"
#include "limits"
#include "math.h"

template<typename T>
class CubicEquation: public ISolve<3, T>
{
public:
	CubicEquation(const c_vector<4, T> &coefficients, const T &epsilon);
	c_vector<3, T> solutions() final;

private:
	Validate<T> validate_;
	c_vector<3, T> solutions_{-1., -1, -1.};
};

template<typename T>
CubicEquation<T>::CubicEquation(const c_vector<4, T> &coefficients, const T &epsilon)
{
	validate_.is_not_zero("cubic_coefficient", coefficients[0], epsilon, "CubicEquation");
	T cubic_coefficient = coefficients[0];
	T quadratic_coefficient = coefficients[1];
	T linear_coefficient = coefficients[2];
	T constant = coefficients[3];


	if (std::abs(constant) < epsilon) {
		solutions_[0] = T{};
		auto quadratic_equation_coefficients =
			c_vector<3, T>{cubic_coefficient, quadratic_coefficient, linear_coefficient};
		auto quadratic_equation = QuadraticEquation<T>(quadratic_equation_coefficients, epsilon);
		for (int i = 0; i < 2; ++i) {
			solutions_[i + 1] = quadratic_equation.solutions()[i];
		}
	}

	auto r_quadratic_coefficient = quadratic_coefficient / cubic_coefficient;
	auto r_linear_coefficient = linear_coefficient / cubic_coefficient;
	auto r_constant = constant / cubic_coefficient;

	auto r_quadratic_coefficient_squared = r_quadratic_coefficient * r_quadratic_coefficient;
	auto p = 1. / 3. * (-r_quadratic_coefficient_squared / 3. + r_linear_coefficient);
	auto q = 1. / 2. * (2. / 27. * r_quadratic_coefficient * r_quadratic_coefficient_squared
		- 1. / 3. * r_quadratic_coefficient * r_linear_coefficient + r_constant);

	auto p_cubed = p * p * p;
	auto discriminant = q * q + p_cubed;

	if (std::abs(discriminant) < epsilon) {
		if (std::abs(q) < epsilon) {
			solutions_[0] = 0;
		}
		else {
			auto q_cube_root = (T)std::cbrt(-q);
			solutions_[0] = 2 * q_cube_root;
			solutions_[1] = -q_cube_root;
		}
	}
	else if (discriminant < 0.) {
		auto phi = 1. / 3. * std::acos(-q / std::sqrt(-p_cubed));
		auto t = 2 * std::sqrt(-p);
		solutions_[0] = t * std::cos(phi);
		std::cout << "PI " <<  t <<"  " << std::cos(phi - M_PI / 3.) <<std::endl;
		solutions_[1] = -t * std::cos(phi + M_PI / 3.);
		solutions_[2] = -t * std::cos(phi - M_PI / 3.);
	}
	else {
		auto root_discriminant = std::sqrt(discriminant);
		solutions_[0] = std::cbrt(root_discriminant - q) - std::cbrt(root_discriminant + q);
	}

}
template<typename T>
c_vector<3, T> CubicEquation<T>::solutions()
{
	return solutions_;
}

#endif //CUBIC_EQUATION_H
