//
// Created by andreas on 11.10.21.
//

#ifndef QUADRATIC_EQUATION_H
#define QUADRATIC_EQUATION_H

#include "interfaces/i_solve.h"
#include "validate.h"
#include "stdexcept"

template <typename T>
class QuadraticEquation : public ISolve
{
public:
	QuadraticEquation(const T & quadratic_factor, const T & linear_factor, const T & constant);
	int number_of_solutions() final;
	void set_epsilon(const double & epsilon) final;
	T get_solution(int number_of_solution);
private:
	T p_;
	T q_;
	double epsilon_{0.00001};
	Validate validate_;
	int number_of_solutions_;
	std::vector<T> solutions_;
};

template<typename T>
QuadraticEquation<T>::QuadraticEquation(const T &quadratic_factor, const T &linear_factor, const T &constant)
{
	validate_.is_above_threshold("quadratic_factor", std::abs(quadratic_factor), epsilon_, "QuadraticEquation");
	p_ = linear_factor/quadratic_factor;
	q_ = constant/quadratic_factor;
	auto root_term = (p_/2.)*(p_/2.) - q_;
	if (root_term < 0)
	{
		number_of_solutions_ = 0;
	}
	else if( std::abs(root_term) < epsilon_)
	{
		number_of_solutions_ = 1;
		auto solution = -p_/2.;
		solutions_.push_back(solution);
	}
	else {
		number_of_solutions_ = 2;
		auto solution1 = -p_/2. + std::sqrt(root_term);
		auto solution2 = -p_/2. - std::sqrt(root_term);
		solutions_.push_back(solution1);
		solutions_.push_back(solution2);

	}

}

template<typename T>
int QuadraticEquation<T>::number_of_solutions()
{
	return number_of_solutions_;
}

template<typename T>
void QuadraticEquation<T>::set_epsilon(const double &epsilon)
{
	epsilon_ = epsilon;
}

template<typename T>
T QuadraticEquation<T>::get_solution(int number_of_solution)
{
	validate_.is_below_threshold("number_of_solution",number_of_solution , 2, "QuadraticEquation")
	validate_.is_above_threshold("number_of_solution",number_of_solution , -1, "QuadraticEquation")
	return solutions_[number_of_solution];
}

#endif //QUADRATIC_EQUATION_H
