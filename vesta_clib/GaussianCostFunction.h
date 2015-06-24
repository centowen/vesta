#include <ceres/ceres.h>
#include <cmath>
using ceres::CostFunction;



#ifndef __GAUSSIAN_COST_FUNCTION_H__
#define __GAUSSIAN_COST_FUNCTION_H__
class GaussianCostFunction : public CostFunction {/*{{{*/
private:
	double _u, _v, _V_real, _V_imag;

public:
	GaussianCostFunction(double u, double v, double V_real, double V_imag)
		: _u(u), _v(v), _V_real(V_real), _V_imag(V_imag)
	{
		for(int i = 0; i < 6; i++) {
			mutable_parameter_block_sizes()->push_back(1);
		}
		set_num_residuals(2);
	}
	virtual ~GaussianCostFunction() {}

	virtual bool Evaluate(double const* const* parameters,
	                      double* residuals,
	                      double** jacobians) const;
};/*}}}*/
#endif
