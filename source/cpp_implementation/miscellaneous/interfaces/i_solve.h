//
// Created by andreas on 11.10.21.
//

#ifndef I_SOLVE_H
#define I_SOLVE_H
template<size_t number_of_solutions, typename T>
class ISolve{
public:
	virtual c_vector<number_of_solutions, T> solutions() = 0;
};
#endif //I_SOLVE_H
