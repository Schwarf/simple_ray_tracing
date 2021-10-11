//
// Created by andreas on 11.10.21.
//

#ifndef I_SOLVE_H
#define I_SOLVE_H

class ISolve{
public:
	virtual int number_of_solutions() = 0;
	virtual void set_epsilon(const double & epsilon) = 0;
};
#endif //I_SOLVE_H
