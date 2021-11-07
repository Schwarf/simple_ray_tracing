//
// Created by andreas on 11.10.21.
//

#ifndef I_SOLVE_H
#define I_SOLVE_H
#include "./../templates/c_vector.h"
template<size_t maximal_number_of_solutions, typename T>
class ISolve{
public:
	virtual c_vector<maximal_number_of_solutions, T> solutions() = 0;
	virtual size_t number_of_solutions() const = 0;
};
#endif //I_SOLVE_H
