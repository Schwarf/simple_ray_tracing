//
// Created by andreas on 07.11.21.
//

#ifndef QUARTIC_EQUATION_H
#define QUARTIC_EQUATION_H
#include "interfaces/i_solve.h"
#include "quadratic_equation.h"
#include "stdexcept"
#include "limits"
#include "cubic_equation.h"
#include "math.h"

template<typename T>
class QuarticEquation: public ISolve<4, T>
{
public:
	QuarticEquation(const FixedSizedArray<5, T> &coefficients, const T &epsilon);
	virtual FixedSizedArray<4, T> solutions() final;
	virtual size_t number_of_solutions() const final;

private:
	FixedSizedArray<4, T> solutions_
		{std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN(),
		 std::numeric_limits<T>::quiet_NaN()};
	size_t number_of_solutions_{};
};


template<typename T>
QuarticEquation<T>::QuarticEquation(const FixedSizedArray<5, T> &coefficients, const T &epsilon)
{
	T absolute_average{};
	for (size_t index = 0; index < 5; index++) {
		absolute_average += std::abs(coefficients[index]);
	}
	absolute_average /= 5.f;

	Validate<float>::is_not_zero("quartic_coefficient", coefficients[0] / absolute_average, epsilon, "QuarticEquation");

	T quartic_coefficient = coefficients[0];
	T cubic_coefficient = coefficients[1];
	T quadratic_coefficient = coefficients[2];
	T linear_coefficient = coefficients[3];
	T constant = coefficients[4];

	auto r_cubic_coefficient = cubic_coefficient / quartic_coefficient;
	auto r_quadratic_coefficient = quadratic_coefficient / quartic_coefficient;
	auto r_linear_coefficient = linear_coefficient / quartic_coefficient;
	auto r_constant = constant / quartic_coefficient;

	auto r_cubic_coefficient_squared = r_cubic_coefficient * r_cubic_coefficient;
	auto p = -3. / 8. * r_cubic_coefficient_squared + r_quadratic_coefficient;
	auto q = 1. / 8 * r_cubic_coefficient_squared * r_cubic_coefficient
		- 1. / 2. * r_cubic_coefficient * r_quadratic_coefficient + r_linear_coefficient;
	auto r = -3. / 256. * r_cubic_coefficient_squared * r_cubic_coefficient_squared
		+ 1. / 16. * r_cubic_coefficient_squared * r_quadratic_coefficient
		- 1. / 4. * r_cubic_coefficient * r_linear_coefficient + r_constant;

	if (std::abs(r) < epsilon) {
		T c_cube = 1.;
		T c_quadratic = 0.;
		T c_linear = p;
		T c_constant = q;
		FixedSizedArray<4, T> cubic_coefficients{c_cube, c_quadratic, c_linear, c_constant};
		auto cubic_solver = CubicEquation<T>(cubic_coefficients, epsilon);
		number_of_solutions_ = cubic_solver.number_of_solutions() + 1;
		solutions_[0] = 0;
		for (size_t index = 0; index < cubic_solver.number_of_solutions(); ++index) {
			solutions_[index + 1] = cubic_solver.solutions()[index];
		}
	}
	else {
		T c_cube = 1.;
		T c_quadratic = -p/ 2.;
		T c_linear = -r;
		T c_constant = r * p/2. - q * q/8.;
		FixedSizedArray<4, T> cubic_coefficients{c_cube, c_quadratic, c_linear, c_constant};
		auto cubic_solver = CubicEquation<T>(cubic_coefficients, epsilon);
		if(cubic_solver.number_of_solutions()  != 1)
		{
			std::cout << "Error in quartic equation. Expect only one real solution in cubic sub-equation!" << std::endl;
		}
		auto z = cubic_solver.solutions()[0];
		auto u = z * z - r;
		auto v = 2 * z - p;
		if (std::abs(u) < epsilon)
		{
			u = 0;
		}
		else if (u > 0) {
			u = std::sqrt(u);
		}
		else
			return;

		if (std::abs(v) < epsilon)
		{
			v = 0;
		}
		else if (v > 0) {
			v = std::sqrt(v);
		}
		else
			return;


		c_quadratic = 1.;
		c_linear = q < 0 ? -v : v;
		c_constant = z - u;
		FixedSizedArray<3, T> quadratic_coefficients{c_quadratic, c_linear, c_constant};
		auto quadratic_solver = QuadraticEquation<T>(quadratic_coefficients, epsilon);
		number_of_solutions_ = quadratic_solver.number_of_solutions();
		for (size_t index = 0; index < quadratic_solver.number_of_solutions(); ++index) {
			solutions_[index] = quadratic_solver.solutions()[index];
		}
		c_quadratic = 1.;
		c_linear = q < 0 ? v : -v;
		c_constant = z + u;
		FixedSizedArray<3, T> quadratic_coefficients2{c_quadratic, c_linear, c_constant};
		auto quadratic_solver2 = QuadraticEquation<T>(quadratic_coefficients2, epsilon);

		for (size_t index = 0; index < quadratic_solver2.number_of_solutions(); ++index) {
			solutions_[number_of_solutions_ + index] = quadratic_solver2.solutions()[index];
		}
		number_of_solutions_ += quadratic_solver2.number_of_solutions();
	}

	for (size_t index = 0; index < number_of_solutions_; ++index) {
		solutions_[index] -= 1. / 4. * r_cubic_coefficient;
	}
}

template<typename T>
FixedSizedArray<4, T> QuarticEquation<T>::solutions()
{
	return solutions_;
}

template<typename T>
size_t QuarticEquation<T>::number_of_solutions() const
{
	return number_of_solutions_;
}


#endif //QUARTIC_EQUATION_H
