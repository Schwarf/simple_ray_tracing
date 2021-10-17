//
// Created by andreas on 11.10.21.
//

#ifndef QUADRATIC_EQUATION_H
#define QUADRATIC_EQUATION_H

#include "interfaces/i_solve.h"
#include "validate.h"
#include "stdexcept"

template<typename T>
class QuadraticEquation: public ISolve
{
public:
	QuadraticEquation(const T &quadratic_coefficient, const T &linear_coefficient, const T &constant);
	size_t number_of_solutions();
	void set_epsilon(const double &epsilon);
	T get_solution(int number_of_solution);
private:
	double epsilon_{0.00001};
	Validate validate_;
	std::vector<T> solutions_;
};

template<typename T>
QuadraticEquation<T>::QuadraticEquation(const T &quadratic_coefficient, const T &linear_coefficient, const T &constant)
{
	validate_
		.is_above_threshold("quadratic_coefficient", std::abs(quadratic_coefficient), epsilon_, "QuadraticEquation");
	// constant is zero: use coefficientization
	if (std::abs(constant) < epsilon_) {
		solutions_.push_back(0.0);
		solutions_.push_back(-linear_coefficient / quadratic_coefficient);
		return;
	}

	auto p = linear_coefficient / quadratic_coefficient;
	auto q = constant / quadratic_coefficient;
	auto discriminant = (p / 2.) * (p / 2.) - q;
	if (discriminant < 0) {
		return;
	}
	if (std::abs(discriminant) < epsilon_) {
		number_of_solutions_ = 1;
		auto solution = -p / 2.;
		solutions_.push_back(solution);
		return;
	}
	number_of_solutions_ = 2;
	auto solution1 = -p / 2. + std::sqrt(discriminant);
	auto solution2 = -p / 2. - std::sqrt(discriminant);
	solutions_.push_back(solution1);
	solutions_.push_back(solution2);

}

template<typename T>
int QuadraticEquation<T>::number_of_solutions()
{
	return solutions_.size();
}

template<typename T>
void QuadraticEquation<T>::set_epsilon(const double &epsilon)
{
	epsilon_ = epsilon;
}

template<typename T>
T QuadraticEquation<T>::get_solution(size_t number_of_solution)
{
	validate_.is_below_threshold("number_of_solution", number_of_solution, 2, "QuadraticEquation")
	validate_.is_above_threshold("number_of_solution", number_of_solution, -1, "QuadraticEquation")
	return solutions_[number_of_solution];
}

#endif //QUADRATIC_EQUATION_H
